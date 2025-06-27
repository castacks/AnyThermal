# loss_combiner.py
import torch
import yaml
from .str_to_loss import str_to_loss_dict
from contextlib import nullcontext

class LossManager:
    def __init__(self, config_path):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.losses = {}
        for name, loss_cfg in self.config["losses"].items():
            loss_cls = str_to_loss_dict[name]
            self.losses[name] = {
                "module": loss_cls(**loss_cfg.get("params", {})),
                "weight": loss_cfg.get("weight", 1.0),
                "report_only": loss_cfg.get("report_only", False),
                "layers": loss_cfg["layers"],
            }
        
        self.layers = []
        for loss_cfg in self.config["losses"].values():
            self.layers.extend(loss_cfg["layers"])
        self.layers = sorted(set(self.layers))

    def compute(self, student_output, teacher_output):
        total_loss = 0.0
        individual_losses = {}
        for name in self.losses.keys():
            for layer in self.layers:
                individual_losses[f"{name}_layer_{layer}"] = 0.0

        for layer in self.layers:
            for name, loss_entry in self.losses.items():
                loss_module = loss_entry["module"]
                report_only = loss_entry["report_only"]
                weight = loss_entry["weight"]
                if layer not in loss_entry["layers"]:
                    continue

                with torch.no_grad() if report_only else nullcontext():
                    loss_value = loss_module(student_output[f"block_{layer}_output"], teacher_output[f"block_{layer}_output"])

                individual_losses[f"{name}_layer_{layer}"] = loss_value.detach().item()
                if not report_only:
                    total_loss += weight * loss_value

        return total_loss, individual_losses
