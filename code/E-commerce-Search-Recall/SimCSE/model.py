import torch
from transformers import AutoModel
import torch.nn.functional as F


class TextBackbone(torch.nn.Module):
    def __init__(self,
                 pretrained='/home/data_ti6_c/wanghk/bert_model/chinese-roberta-wwm-ext',
                 output_dim=128) -> None:
        super(TextBackbone, self).__init__()
        self.extractor = AutoModel.from_pretrained(pretrained).cuda()
        self.drop = torch.nn.Dropout(p=0.2)
        self.fc = torch.nn.Linear(768, output_dim)

    def forward(self, input_ids, attention_mask, token_type_ids):
        x = self.extractor(input_ids,
                           attention_mask=attention_mask,
                           token_type_ids=token_type_ids,
                            output_hidden_states = True
                           )
        first = x.hidden_states[1].transpose(1, 2)
        last = x.hidden_states[-1].transpose(1, 2)
        first_avg = torch.avg_pool1d(first, kernel_size=first.shape[-1]).squeeze(-1)  # [batch, 768]
        last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
        avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)  # [batch, 2, 768]
        out = torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)
        out = self.drop(out)
        out = self.fc(out)
        out = F.normalize(out, p=2, dim=-1)
        return out

    def predict(self, x):
        x["input_ids"] = x["input_ids"].squeeze(1)
        x["attention_mask"] = x["attention_mask"].squeeze(1)
        x["token_type_ids"] = x["token_type_ids"].squeeze(1)
        x["output_hidden_states"] = True

        x = self.extractor(**x)
        first = x.hidden_states[1].transpose(1, 2)
        last = x.hidden_states[-1].transpose(1, 2)
        first_avg = torch.avg_pool1d(first, kernel_size=first.shape[-1]).squeeze(-1)  # [batch, 768]
        last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
        avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)  # [batch, 2, 768]
        out = torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)

        out = self.fc(out)
        out = F.normalize(out, p=2, dim=-1)

        return out
