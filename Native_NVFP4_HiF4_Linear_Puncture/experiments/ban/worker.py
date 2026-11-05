"""Explicit worker installed only for Ban or its calibration."""
from vllm.v1.worker.gpu_worker import Worker as VllmWorker


class Worker(VllmWorker):
    def load_model(self, *, load_dummy_weights=False):
        from .runtime import BanState, install_dispatch, install_model
        super().load_model(load_dummy_weights=load_dummy_weights)
        self.ban_state = BanState(self.vllm_config.additional_config["ban"], self.device)
        install_model(self.model_runner.model, self.ban_state)
        install_dispatch(self.ban_state)

    def ban_configure(self, mode=None, layer=None, k_pruned=3, capture=None, execution=None,
                      lam=None, k_min=None, topn=1000, probe=False):
        from .algorithm import validate_parameters
        state = self.ban_state
        mode = state.mode if mode is None else mode
        if mode not in {"native", "collect", "identity", "full", "ban", "single_layer"}:
            raise ValueError("unknown Ban mode")
        if mode == "ban" and state.sensitivity is None:
            raise ValueError("Ban requires a matching calibration artifact")
        if mode == "single_layer" and (layer not in range(state.layers) or not 1 <= k_pruned <= state.top_k):
            raise ValueError("invalid calibration layer/k")
        execution = state.execution if execution is None else execution
        if execution not in {"masked", "compact"}:
            raise ValueError("unknown expert execution mode")
        lam = state.config.get("lam", 0.7) if lam is None else lam
        k_min = state.config.get("k_min", 3) if k_min is None else k_min
        validate_parameters(lam, k_min, state.top_k)
        state.config.update(lam=lam, k_min=k_min)
        state.mode, state.layer, state.k_pruned = mode, layer, k_pruned
        state.capture, state.execution, state.topn, state.probe = capture, execution, topn, probe
        state.reset()
        state.active = True
        return {"mode": mode, "layers": state.layers, "top_k": state.top_k}

    def ban_snapshot(self):
        return self.ban_state.snapshot()

    def ban_load_artifact(self, path):
        self.ban_state.load(path)
        return {"loaded": True}

    def ban_context(self):
        from vllm.model_executor.layers.quantization.hif4_runtime import current_hif4_runtime_spec
        explicit = current_hif4_runtime_spec(self.ban_state.config["spec_path"])
        implicit = current_hif4_runtime_spec()
        if implicit is None or implicit is not explicit:
            raise RuntimeError("implicit/explicit HiF4 sidecar resolution differs")
        return {"variant": implicit["variant"], "use_r64": implicit["use_r64"]}
