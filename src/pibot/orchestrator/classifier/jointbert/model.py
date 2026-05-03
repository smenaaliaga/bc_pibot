"""
JointBERT model for joint intent classification and slot filling.
Based on https://github.com/monologg/JointBERT
"""
import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import ModelOutput
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class JointBERTOutput(ModelOutput):
    """Output of JointBERT model"""
    loss: Optional[torch.FloatTensor] = None
    intent_logits: torch.FloatTensor = None
    slot_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class IntentClassifier(nn.Module):
    """Intent classification head"""
    def __init__(self, input_dim, num_intent_labels, dropout_rate=0.):
        super(IntentClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_intent_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)


class SlotClassifier(nn.Module):
    """Slot classification head"""
    def __init__(self, input_dim, num_slot_labels, dropout_rate=0.):
        super(SlotClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_slot_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)


class JointBERT(BertPreTrainedModel):
    """
    Joint BERT model for intent classification and slot filling.
    
    Args:
        config: BERT configuration
        args: Training arguments with intent_label_lst, slot_label_lst, use_crf, etc.
        intent_label_lst: List of intent labels
        slot_label_lst: List of slot labels (BIO tagging)
    """
    
    def __init__(self, config, args=None, intent_label_lst=None, slot_label_lst=None):
        super(JointBERT, self).__init__(config)
        self.args = args
        self.num_intent_labels = len(intent_label_lst) if intent_label_lst else config.num_labels
        self.num_slot_labels = len(slot_label_lst) if slot_label_lst else config.num_labels
        
        # BERT encoder
        self.bert = BertModel(config)
        
        # Classification heads
        self.intent_classifier = IntentClassifier(
            config.hidden_size,
            self.num_intent_labels,
            getattr(args, 'dropout_rate', 0.1) if args else 0.1
        )
        
        self.slot_classifier = SlotClassifier(
            config.hidden_size,
            self.num_slot_labels,
            getattr(args, 'dropout_rate', 0.1) if args else 0.1
        )
        
        # CRF layer (optional)
        self.use_crf = getattr(args, 'use_crf', False) if args else False
        if self.use_crf:
            from torchcrf import CRF
            self.crf = CRF(num_tags=self.num_slot_labels, batch_first=True)
        
        # Loss weights
        self.intent_loss_weight = getattr(args, 'intent_loss_coef', 1.0) if args else 1.0
        self.slot_loss_weight = getattr(args, 'slot_loss_coef', 1.0) if args else 1.0
        
        # Initialize weights
        self.post_init()
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        intent_label_ids=None,
        slot_labels_ids=None,
        **kwargs
    ):
        """
        Forward pass.
        
        Returns:
            If intent_label_ids and slot_labels_ids are provided: (loss, (intent_logits, slot_logits))
            Otherwise: (intent_logits, slot_logits)
        """
        # BERT encoding
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        sequence_output = outputs[0]  # (batch_size, seq_len, hidden_size)
        pooled_output = outputs[1]    # (batch_size, hidden_size) - [CLS] token
        
        # Intent classification
        intent_logits = self.intent_classifier(pooled_output)
        
        # Slot classification
        slot_logits = self.slot_classifier(sequence_output)
        
        # Calculate loss if labels are provided
        total_loss = 0
        if intent_label_ids is not None and slot_labels_ids is not None:
            # Intent loss
            intent_loss_fct = nn.CrossEntropyLoss()
            intent_loss = intent_loss_fct(
                intent_logits.view(-1, self.num_intent_labels),
                intent_label_ids.view(-1)
            )
            
            # Slot loss
            if self.use_crf:
                # CRF loss
                slot_loss = -self.crf(
                    slot_logits,
                    slot_labels_ids,
                    mask=attention_mask.byte(),
                    reduction='mean'
                )
            else:
                # Cross entropy loss
                slot_loss_fct = nn.CrossEntropyLoss(ignore_index=0)  # 0 is padding
                # Only consider active slots (not padding)
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
                    active_labels = slot_labels_ids.view(-1)[active_loss]
                    slot_loss = slot_loss_fct(active_logits, active_labels)
                else:
                    slot_loss = slot_loss_fct(
                        slot_logits.view(-1, self.num_slot_labels),
                        slot_labels_ids.view(-1)
                    )
            
            # Combined loss
            total_loss = self.intent_loss_weight * intent_loss + self.slot_loss_weight * slot_loss
        
        # Return format depends on whether we're training or inferring
        if intent_label_ids is not None and slot_labels_ids is not None:
            return total_loss, (intent_logits, slot_logits)
        else:
            return intent_logits, slot_logits
