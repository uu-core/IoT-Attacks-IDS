import torch
import numpy as np
import logging
import evaluation as evaluate
from utils import confidence_from_logits

def eval_model(args, model,test_domain_loader, train_domain, device, domain_id=None):
    all_y_true, all_y_pred, all_y_prob = [], [], []
    all_confidences, all_conf_correct, all_conf_incorrect = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in test_domain_loader[train_domain]:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            if domain_id is not None:
                outputs, _ = model(X_batch, domain_id=domain_id)  # Pass domain_id to the model
            else:
                outputs, _ = model(X_batch)  # Pass domain_id to the model
            
            # === confidence here (eval only) ===
            probs, preds, confs = confidence_from_logits(outputs)
            all_y_true.extend(y_batch.cpu().numpy())
            all_y_pred.extend(preds.cpu().numpy())
            all_y_prob.extend(probs[:, 1].cpu().numpy())   # you already use class-1 prob for ROC
            all_confidences.extend(confs.cpu().numpy())

            correct_mask = (preds == y_batch)
            if correct_mask.any():
                all_conf_correct.extend(confs[correct_mask].cpu().numpy())
            if (~correct_mask).any():
                all_conf_incorrect.extend(confs[~correct_mask].cpu().numpy())
            _, predicted = torch.max(outputs.data, 1)
            all_y_true.extend(y_batch.cpu().numpy())
            all_y_pred.extend(predicted.cpu().numpy())
            all_y_prob.extend(torch.nn.functional.softmax(outputs, dim=1)[:, 1].cpu().numpy())
        
    metrics = evaluate.evaluate_metrics(np.array(all_y_true), np.array(all_y_pred),
                        np.array(all_y_prob), train_domain, train_domain)
    
    avg_conf           = float(np.mean(all_confidences))   if all_confidences else float("nan")
    avg_conf_correct   = float(np.mean(all_conf_correct))  if all_conf_correct else float("nan")
    avg_conf_incorrect = float(np.mean(all_conf_incorrect))if all_conf_incorrect else float("nan")
    metrics["avg_conf"] = avg_conf
    metrics["avg_conf_correct"] = avg_conf_correct
    metrics["avg_conf_incorrect"] = avg_conf_incorrect
    
    return metrics