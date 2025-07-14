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
        self.get_second_view = False
        for name, loss_cfg in self.config["losses"].items():
            loss_cls = str_to_loss_dict[loss_cfg["type"]]
            self.losses[name] = {
                "module": loss_cls(**loss_cfg.get("params", {})),
                "weight": loss_cfg.get("weight", 1.0),
                "report_only": loss_cfg.get("report_only", False),
                "layers": loss_cfg["layers"],
            }
            if hasattr(self.losses[name]["module"], "second_view"):
                self.get_second_view = True
        
        self.layers = []
        for loss_cfg in self.config["losses"].values():
            for layer in loss_cfg["layers"]:
                if layer not in self.layers:
                    self.layers.append(layer)


    def compute(self,student_output, second_view_student_output, teacher_output):
        assert not (self.get_second_view and second_view_student_output is None), "Second view student output is required for losses that use it."
        if self.get_second_view ==False:
            assert second_view_student_output is None, "Second view student output should be None for losses that do not use it."
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
                    if hasattr(loss_module, "second_view"):
                        input1_list = [student_output[f"block_{layer}_output"]]
                        if f"block_{layer}_shared_output" in student_output:
                            input1_list += [student_output[f"block_{layer}_shared_output"],student_output[f"block_{layer}_specific_output"]]
                        input2_list = [second_view_student_output[f"block_{layer}_output"]]
                        if f"block_{layer}_shared_output" in second_view_student_output:
                            input2_list += [second_view_student_output[f"block_{layer}_shared_output"],second_view_student_output[f"block_{layer}_specific_output"]]  
                        loss_value = loss_module(input1_list, input2_list)
                    else:
                        input1_list = [student_output[f"block_{layer}_output"]]
                        if f"block_{layer}_shared_output" in student_output:
                            input1_list += [student_output[f"block_{layer}_shared_output"],student_output[f"block_{layer}_specific_output"]]
                        input2_list = [teacher_output[f"block_{layer}_output"]]
                        if f"block_{layer}_shared_output" in teacher_output:
                            input2_list += [teacher_output[f"block_{layer}_shared_output"],teacher_output[f"block_{layer}_specific_output"]]
                        loss_value = loss_module(input1_list, input2_list)


                individual_losses[f"{name}_layer_{layer}"] = loss_value.detach().cpu().item()
                if not report_only:
                    total_loss += weight * loss_value

        return total_loss, individual_losses
