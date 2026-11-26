"""Bounded-memory SSD for complete, unpadded calibration sequences."""

import torch


def calibration_forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
    # Calibration processes one complete, unpadded sequence. Keep Linear calls
    # visible to input collection and W4A4, including the output projection.
    if cache_params is not None or hidden_states.shape[0] != 1 or self.training:
        raise ValueError("Mamba calibration requires one full sequence in eval mode without a cache")
    if attention_mask is not None and not bool(torch.all(attention_mask == 1)):
        raise ValueError("Calibration sequences must contain only valid tokens")
    from transformers.models.nemotron_h.modeling_nemotron_h import segment_sum

    length = hidden_states.shape[1]
    projected = self.in_proj(hidden_states)
    gate, mixed, dt = projected.split([self.intermediate_size, self.conv_dim, self.num_heads], -1)
    mixed = self.act(self.conv1d(mixed.transpose(1, 2)).transpose(1, 2)[:, :length])
    group_width = self.n_groups * self.ssm_state_size
    x, b, c = mixed.split([self.intermediate_size, group_width, group_width], -1)
    # Match Transformers' reference arithmetic: dt is rounded in the source
    # dtype before the FP32 scan; normalization sees the FP32 scan output.
    dt = torch.nn.functional.softplus(dt + self.dt_bias).clamp_min(self.time_step_min)
    x = x.reshape(length, self.num_heads, self.head_dim).float()
    b = b.reshape(length, self.n_groups, self.ssm_state_size).float()
    c = c.reshape_as(b).float()
    out = torch.empty_like(x)
    a = -torch.exp(self.A_log.float())
    state = x.new_zeros(self.num_heads, self.head_dim, self.ssm_state_size)
    # Contract the state dimension before materializing intra-chunk attention.
    # The reference broadcasts [chunks,L,L,heads,state], which can exceed
    # 150 GiB. Carry the exact recurrent state across chunks; do not truncate
    # the calibration sequence or reset state at chunk boundaries.
    for start in range(0, length, self.chunk_size):
        end = min(start + self.chunk_size, length)
        xc = x[start:end].transpose(0, 1)
        bc = b[start:end].repeat_interleave(self.num_heads // self.n_groups, 1).transpose(0, 1)
        cc = c[start:end].repeat_interleave(self.num_heads // self.n_groups, 1).transpose(0, 1)
        dc = dt[0, start:end].T.float()
        ac = dc * a[:, None]
        cumulative = ac.cumsum(-1)
        decay = segment_sum(ac).exp()
        xd = xc * dc[..., None]
        intra = torch.bmm(cc, bc.transpose(1, 2)) * decay
        y = torch.bmm(intra, xd)
        y += torch.bmm(cc, state.transpose(1, 2)) * cumulative.exp()[..., None]
        y += xc * self.D[:, None, None]
        out[start:end] = y.transpose(0, 1)
        decay_to_end = (cumulative[:, -1:] - cumulative).exp()
        state = (state * cumulative[:, -1].exp()[:, None, None]
                 + torch.bmm(xd.transpose(1, 2), bc * decay_to_end[..., None]))
    return self.out_proj(self.norm(out.reshape(1, length, -1), gate).to(hidden_states.dtype))
