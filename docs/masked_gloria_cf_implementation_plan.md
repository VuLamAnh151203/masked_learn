# Implementation Plan: MASKED_GLORIA_CF

## 1. Mục tiêu

Tạo một bản sao thực nghiệm mới từ `src/models/masked_gloria.py` với tên:

- File model: `src/models/masked_gloria_cf.py`
- Class model: `MASKED_GLORIA_CF`

Experiment cần kiểm tra **Boundary-aware Counterfactual Robustness**:

```text
L = L_rec + lambda_cf * L_boundary
```

Trong đó gradient phải được kiểm soát như sau:

```text
L_rec       -> theta, mask_logits
L_boundary  -> theta only
```

Nghĩa là CF branch phải dùng mask đã detach:

```python
mask_cf = self.get_forward_edge_mask().detach()
```

Mục tiêu của Exp A là kiểm tra representation robustness khi graph bị counterfactual perturbation, chưa dùng boundary loss để supervise mask.

---

## 2. Trạng thái code đã xác nhận

Từ `src/models/masked_gloria.py`:

- Model hiện tại là `MASKED_GLORIA`.
- Learnable edge mask là:

```python
self.mask_logits = nn.Parameter(torch.zeros(self.num_interactions, device=self.device))
```

- Effective mask được lấy bằng:

```python
torch.sigmoid(self.mask_logits)
```

- `forward_edge_index` lưu các cạnh user-item theo chiều forward:

```python
self.forward_edge_index = edge_index
self.forward_edge_users = edge_index[0]
self.forward_edge_items = edge_index[1] - self.num_user
```

- Graph dùng cho GCN là graph hai chiều:

```python
self.edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)
```

- Khi truyền mask vào masked GCN, mask forward được nhân đôi cho chiều reverse:

```python
edge_mask = torch.cat([forward_edge_mask, forward_edge_mask], dim=0)
```

- Hàm build embedding chính:

```python
compute_result_embedding(forward_edge_mask=None, full_view=None)
```

- Hàm loss baseline:

```python
calculate_loss(interaction)
```

hiện trả về BPR-style loss:

```python
-pos_mean(log2(sigmoid(pos_scores - neg_scores)))
```

Từ `src/common/trainer.py`:

- Trainer gọi `model.set_training_epoch(epoch_idx)` nếu model có hook này.
- Trainer gọi `model.pre_epoch_processing()` trước mỗi epoch.
- Trainer gọi `model.post_epoch_processing()` sau mỗi epoch.
- Train loop dùng `loss_func(interaction)` và gọi `loss.backward()` một lần.

=> Có thể implement Exp A chủ yếu trong model mới, không cần custom trainer ở V1.

---

## 3. Phạm vi thay đổi

### Bắt buộc

- Tạo `src/models/masked_gloria_cf.py` bằng cách copy logic từ `masked_gloria.py`.
- Đổi class thành `MASKED_GLORIA_CF`.
- Thêm config cho CF branch.
- Thêm hook epoch để warm-up.
- Thêm auxiliary boundary loss trong `calculate_loss`.
- Đảm bảo `L_boundary` không update `mask_logits`.

### Có thể cần

- Thêm CLI args trong `src/main.py` để override config CF dễ hơn.
- Không cần sửa `src/utils/utils.py` vì `get_model()` tự import theo tên model lower-case:

```text
MASKED_GLORIA_CF -> models.masked_gloria_cf -> class MASKED_GLORIA_CF
```

### Không làm ở V1

- Không sửa `src/models/masked_gloria.py`.
- Không thêm custom trainer nếu model hook đã đủ.
- Không thay đổi data split.
- Không dùng validation/test interaction trong CF branch.

---

## 4. Config đề xuất

Thêm default trong `MASKED_GLORIA_CF.__init__` bằng cách đọc `config` với fallback:

```yaml
cf_lambda: 0.1
cf_warmup_ratio: 0.10
cf_warmup_epochs: -1
cf_user_ratio: 0.10
cf_batch_size: 8
cf_k: 20
cf_boundary_width: 5
cf_boundary_q: 3
cf_temperature: 1.0
cf_min_history: 2
cf_drop_bidirectional: true
cf_seed_offset: 10000
cf_log_stats: true
```

Ý nghĩa:

- `cf_lambda`: trọng số của boundary loss.
- `cf_warmup_ratio`: warm-up theo phần trăm tổng epoch.
- `cf_warmup_epochs`: nếu > 0 thì override `cf_warmup_ratio`.
- `cf_user_ratio`: tỉ lệ user sample cho auxiliary branch mỗi epoch.
- `cf_batch_size`: số user xử lý trong CF branch mỗi batch train.
- `cf_k`: cutoff Top-K để detect fragile.
- `cf_boundary_width`: vùng sát mép trong Top-K, ví dụ K=20 width=5 -> rank 16..20.
- `cf_boundary_q`: số competitor ngoài/cạnh cutoff, lấy rank K..K+q.
- `cf_temperature`: hệ số T trong softplus.
- `cf_min_history`: user phải có ít nhất 2 training edges để bỏ `p` và `e`.
- `cf_drop_bidirectional`: giữ true vì graph hiện tại lưu hai chiều bằng cách nhân đôi mask.
- `cf_seed_offset`: giúp sample CF ổn định theo epoch nhưng khác seed chính.
- `cf_log_stats`: log số fragile/sample/loss mỗi epoch.

---

## 5. Công thức loss

Main loss giữ nguyên:

```python
loss_rec = baseline_bpr_loss(interaction)
```

Boundary loss cho user `u`:

```text
L_boundary(u) = mean_j softplus(T * (s_cf[u, j] - s_cf[u, p]))
```

với:

```text
j in B_u
B_u = items at ranks K, K+1, ..., K+q from probe ranking
```

Tổng loss:

```python
loss = loss_rec + self.cf_lambda * loss_boundary
```

Trong warm-up:

```python
loss = loss_rec
```

Nếu epoch đó không có fragile sample hợp lệ:

```python
loss_boundary = 0.0
loss = loss_rec
```

---

## 6. Sơ đồ pipeline

```text
Training batch
    |
    |-- Main branch
    |      G with learned mask
    |      loss_rec
    |      gradient -> theta + mask_logits
    |
    |-- Auxiliary branch, after warm-up only
           sample subset users
           choose pseudo-positive p=(u, i_p)
           G_probe = G \ {p}
           rank i_p under G_probe
           if rank near Top-K boundary:
               choose another history edge e=(u, i_e), e != p
               G_cf = G \ {p, e}
               choose boundary competitors B_u from probe ranking
               compute pairwise softplus boundary loss
               gradient -> theta only
```

---

## 7. Phase 1 - Tạo model copy

Checklist:

- [ ] Copy `src/models/masked_gloria.py` sang `src/models/masked_gloria_cf.py`.
- [ ] Đổi class:

```python
class MASKED_GLORIA_CF(GeneralRecommender):
```

- [ ] Đổi super call:

```python
super(MASKED_GLORIA_CF, self).__init__(config, dataset)
```

- [ ] Giữ nguyên baseline behavior khi `cf_lambda = 0`.
- [ ] Không làm thay đổi checkpoint/state_dict của model gốc.

Acceptance:

- Chạy được import model:

```powershell
python src/main.py --model MASKED_GLORIA_CF --dataset book
```

với CF có thể tắt bằng `cf_lambda=0`.

---

## 8. Phase 2 - Build training history và edge mapping

Trong `__init__`, tận dụng các tensor đã có:

```python
self.forward_edge_users
self.forward_edge_items
```

Cần tạo cấu trúc phục vụ CF:

```python
self.user_to_edge_ids: List[Tensor]
self.user_seen_items: List[Tensor]
```

Logic:

- Với mỗi `edge_id`, lấy:

```python
u = self.forward_edge_users[edge_id]
i = self.forward_edge_items[edge_id]
```

- Append `edge_id` vào history của user `u`.
- Append `i` vào seen items của user `u`.
- Chỉ dùng training interactions vì model nhận `train_data` làm dataset khi khởi tạo.

Pseudo-positive chọn từ history:

- Nếu dataset có timestamp trong tương lai thì ưu tiên edge muộn nhất.
- Code hiện tại `RecDataset.load_inter_graph()` chỉ đọc user, item, split label; chưa thấy timestamp.
- Vì vậy V1 chọn random một edge trong training history.

Điều kiện hợp lệ:

```python
len(user_to_edge_ids[u]) >= cf_min_history
```

Với Exp A cần tối thiểu 2 edges để drop `p` và `e`.

---

## 9. Phase 3 - Warm-up và user sampling

Thêm hook:

```python
def set_training_epoch(self, epoch_idx):
    self.current_epoch = int(epoch_idx)
```

Tính warm-up:

```python
if cf_warmup_epochs > 0:
    warmup_epochs = cf_warmup_epochs
else:
    warmup_epochs = int(math.ceil(config['epochs'] * cf_warmup_ratio))
```

Trong `calculate_loss`:

```python
loss_rec = self.calculate_rec_loss(interaction)

if self.current_epoch < self.cf_warmup_epochs:
    return loss_rec

loss_boundary = self.calculate_boundary_loss()
return loss_rec + self.cf_lambda * loss_boundary
```

Sampling user:

- Main branch vẫn dùng 100% batch từ dataloader như hiện tại.
- Auxiliary branch sample user độc lập mỗi batch hoặc mỗi epoch.
- V1 nên sample theo batch train để đơn giản:

```python
num_cf_users = max(1, int(num_users_in_batch * cf_user_ratio))
```

Nhưng do user trong interaction batch có thể trùng, nên lấy unique:

```python
candidate_users = torch.unique(interaction[0])
```

Nếu muốn đúng 10-20% toàn user mỗi epoch, có thể pre-sample trong `pre_epoch_processing()`, nhưng V1 per-batch nhẹ hơn và ít sửa trainer hơn.

Recommendation:

- V1: sample từ `torch.unique(interaction[0])` để chi phí thấp.
- V2: sample global subset trong `pre_epoch_processing()` nếu cần đúng theo mô tả paper.

---

## 10. Phase 4 - Probe graph `G_probe = G \ {p}`

Vì graph hiện tại sử dụng `edge_index` cố định và mask theo forward edge, không cần rebuild graph.

Tạo probe mask:

```python
base_mask = self.get_forward_edge_mask().detach()
probe_mask = base_mask.clone()
probe_mask[p_edge_id] = 0.0
```

Sau đó:

```python
with torch.no_grad():
    probe_embed = self.compute_result_embedding(
        forward_edge_mask=probe_mask,
        full_view=full_view
    )
```

Lưu ý:

- Probe branch chỉ dùng để chọn fragile case và boundary competitors.
- Vì chỉ selection, dùng `torch.no_grad()` là đúng.
- Dùng `base_mask.detach()` để selection không tạo graph gradient.

Ranking:

```python
user_vec = probe_embed[u]
item_mat = probe_embed[self.num_user:]
scores = torch.matmul(item_mat, user_vec)
```

Mask seen items:

```python
remaining_seen = seen_items[u] excluding i_p
scores[remaining_seen] = -1e10
```

Không mask `i_p`, vì cần tính rank pseudo-positive sau khi remove `p` khỏi graph.

---

## 11. Phase 5 - Fragile detection

Rank dùng 1-based rank.

```python
rank_p = 1 + torch.sum(scores > scores[i_p])
```

Fragile nếu:

```python
lower = cf_k - cf_boundary_width + 1
upper = cf_k
is_fragile = lower <= rank_p <= upper
```

Ví dụ:

```text
K=20, width=5 -> rank 16..20
K=5,  width=2 -> rank 4..5
```

Nếu không fragile:

- bỏ sample đó;
- không tạo CF branch;
- không tính loss cho user đó.

---

## 12. Phase 6 - Boundary competitor selection

Từ probe ranking, lấy item quanh cutoff:

```text
B_u = items at ranks K, K+1, ..., K+q
```

Implementation:

```python
top_count = cf_k + cf_boundary_q
_, ranked_items = torch.topk(scores, k=top_count)
boundary_items = ranked_items[cf_k - 1 : cf_k + cf_boundary_q]
```

Cần lọc:

- bỏ `i_p` khỏi `boundary_items`;
- bỏ item invalid nếu bị mask score `-1e10`;
- nếu `boundary_items` rỗng thì skip sample.

Boundary IDs phải stop-gradient:

```python
boundary_items = boundary_items.detach()
```

Vì item IDs không có gradient, detach chủ yếu để thể hiện rõ selection không thuộc computational graph.

---

## 13. Phase 7 - CF graph `G_cf = G \ {p, e}`

Chọn một history edge khác:

```python
candidate_e = user_to_edge_ids[u] excluding p_edge_id
sample e_edge_id from candidate_e
```

Tạo CF mask:

```python
mask_cf = self.get_forward_edge_mask().detach().clone()
mask_cf[p_edge_id] = 0.0
mask_cf[e_edge_id] = 0.0
```

Không cần remove reverse edge thủ công vì `compute_result_embedding()` tự nhân đôi forward mask:

```python
edge_mask = torch.cat([mask_cf, mask_cf], dim=0)
```

Như vậy `p` và `e` đều bị drop cả hai chiều.

Quan trọng:

- CF forward **không** dùng `torch.no_grad()`.
- `mask_cf` detach nên boundary loss không update `mask_logits`.
- Embedding/GNN parameters vẫn có gradient.

---

## 14. Phase 8 - Boundary loss implementation

Tính embedding CF:

```python
cf_embed = self.compute_result_embedding(
    forward_edge_mask=mask_cf,
    full_view=full_view
)
```

Tính scores:

```python
user_vec = cf_embed[u]
item_mat = cf_embed[self.num_user:]
score_p = torch.sum(user_vec * item_mat[i_p], dim=-1)
score_boundary = torch.matmul(item_mat[boundary_items], user_vec)
```

Loss:

```python
loss_u = F.softplus(
    self.cf_temperature * (score_boundary - score_p)
).mean()
```

Aggregate:

```python
loss_boundary = torch.stack(losses).mean()
```

Nếu không có loss hợp lệ:

```python
loss_boundary = loss_rec.new_tensor(0.0)
```

Return nên là tuple để trainer log được từng phần:

```python
return loss_rec, self.cf_lambda * loss_boundary
```

hoặc return scalar:

```python
return loss_rec + self.cf_lambda * loss_boundary
```

Recommendation:

- Dùng tuple `(loss_rec, cf_weighted_loss)` để log `train_loss1`, `train_loss2`.
- Đảm bảo trainer hiện tại sẽ sum tuple trước backward, đã được hỗ trợ.

---

## 15. Phase 9 - Tránh side effect của `result_embed`

`compute_result_embedding()` hiện set:

```python
self.result_embed = ...
```

Nếu gọi probe/CF branch sau main branch, `self.result_embed` có thể bị ghi đè bởi embedding counterfactual.

Cách xử lý V1:

- Trong `calculate_loss`, sau khi tính CF xong, gọi lại:

```python
self.compute_result_embedding()
```

để restore embedding training graph cho evaluation sau đó.

Cách tốt hơn V2:

- Tách helper build embedding không ghi state:

```python
_build_result_embedding(forward_edge_mask, full_view)
```

- `compute_result_embedding()` chỉ wrapper và cache vào `self.result_embed`.

Recommendation:

- Nếu muốn ít sửa nhất: V1 restore `self.compute_result_embedding()` cuối `calculate_loss` khi đã chạy CF.
- Nếu muốn sạch hơn: refactor helper pure function trong `masked_gloria_cf.py`.

---

## 16. Phase 10 - Logging

Thêm counters reset trong `pre_epoch_processing()`:

```python
self.cf_stats = {
    'samples': 0,
    'eligible': 0,
    'fragile': 0,
    'used': 0,
    'loss_sum': 0.0,
}
```

Update trong boundary branch:

- `samples`: số user được sample.
- `eligible`: số user đủ history.
- `fragile`: số user có pseudo-positive sát boundary.
- `used`: số user thật sự đóng góp loss.
- `loss_sum`: tổng unweighted boundary loss.

Thêm `post_epoch_processing()`:

```python
def post_epoch_processing(self):
    if not self.cf_log_stats:
        return None
    return 'cf samples: ..., fragile: ..., used: ..., boundary_loss: ...'
```

Nếu copy từ model gốc mà chưa có `pre_epoch_processing/post_epoch_processing`, cần implement cả hai để trainer gọi không lỗi.

---

## 17. Phase 11 - CLI integration

Hiện `src/main.py` có parser cho các experiment EX2/EX3. Với CF có thể thêm args:

```powershell
--cf_lambda
--cf_warmup_ratio
--cf_warmup_epochs
--cf_user_ratio
--cf_batch_size
--cf_k
--cf_boundary_width
--cf_boundary_q
--cf_temperature
--cf_min_history
```

Và trong `config_dict`:

```python
elif args.model.upper() == 'MASKED_GLORIA_CF':
    config_dict.update({
        'cf_lambda': args.cf_lambda,
        ...
    })
```

Nếu muốn implement nhanh nhất:

- Không cần sửa CLI ban đầu.
- Đặt fallback config trong model.
- Sau đó thêm CLI args khi cần sweep hyperparameter.

---

## 18. Phase 12 - Test plan

### Smoke test import

```powershell
python src/main.py --model MASKED_GLORIA_CF --dataset book
```

### Baseline equivalence

Chạy với:

```yaml
cf_lambda: 0.0
```

Kỳ vọng:

- Loss gần giống `MASKED_GLORIA` cùng seed.
- Không chạy boundary branch hoặc branch trả loss 0.

### Warm-up test

Set:

```yaml
cf_warmup_epochs: 2
```

Kỳ vọng:

- Epoch 0-1 chỉ có `loss_rec`.
- Sau epoch 2 mới có stats CF.

### Gradient isolation test

Sau backward với CF active, kiểm tra:

```python
mask_logits.grad_from_boundary == 0
```

Cách thực tế:

- Chạy một batch chỉ `loss_boundary.backward()`.
- Assert `self.mask_logits.grad is None` hoặc toàn zero.
- Các parameter như `id_embedding_masked.weight` có grad.

### Fragile selection test nhỏ

Dùng batch nhỏ, log:

- sampled users;
- rank pseudo-positive;
- boundary items;
- số sample skip.

Kỳ vọng:

- `i_p` không bị mask khi rank.
- Remaining history bị mask khỏi candidate ranking.
- `B_u` không chứa `i_p`.

---

## 19. Rủi ro và quyết định cần benchmark

### Hiệu năng

Mỗi fragile user cần thêm forward GCN trên graph CF. Nếu sample quá nhiều user, training sẽ rất chậm.

Giảm chi phí bằng:

- `cf_user_ratio=0.05..0.10`;
- `cf_batch_size` nhỏ;
- chỉ chạy CF sau warm-up;
- skip sớm user không fragile bằng probe `no_grad`.

### Normalization

`Base_gcn.message()` tính degree từ `edge_index`, không tính lại degree theo mask. Khi set mask edge về 0, normalization vẫn theo graph gốc.

V1 recommendation:

- Giữ fixed normalization để thay đổi tối thiểu và nhất quán với current mask mechanism.

V2 option:

- Recompute degree theo active edge mask nếu muốn graph removal đúng toán học hơn.
- Nhưng việc này thay đổi semantics của model gốc và cần benchmark riêng.

### Cache side effect

`compute_result_embedding()` ghi `self.result_embed`. Probe/CF có thể làm evaluation dùng nhầm embedding CF nếu không restore.

Cần xử lý như Phase 9.

### Timestamp

Code hiện tại không load timestamp. Vì vậy V1 không thể ưu tiên interaction muộn nhất.

Nếu cần đúng mô tả:

- Sửa `RecDataset.load_inter_graph()` để đọc timestamp column.
- Lưu timestamp trong train dataset.
- Sort user history theo timestamp.

Không khuyến nghị làm ngay nếu dataset hiện tại không thống nhất timestamp.

---

## 20. Tiêu chí nghiệm thu

Implementation được coi là đạt nếu:

- [ ] `MASKED_GLORIA_CF` chạy được qua existing `Trainer`.
- [ ] `L_rec` vẫn update cả `theta` và `mask_logits`.
- [ ] `L_boundary` chỉ update `theta`, không update `mask_logits`.
- [ ] Warm-up hoạt động đúng số epoch.
- [ ] Auxiliary branch chỉ dùng training history.
- [ ] Probe ranking không mask pseudo-positive `i_p`.
- [ ] Remaining seen items được mask khỏi candidate ranking.
- [ ] Fragile detection dùng rank, không dùng raw score margin.
- [ ] Boundary competitors lấy quanh cutoff `K..K+q` và không chứa `i_p`.
- [ ] Drop interaction trong graph hai chiều được xử lý bằng forward mask nhân đôi.
- [ ] Có logging số sample, fragile, used và boundary loss.

---

## 21. Lệnh chạy dự kiến sau khi implement

Chạy mặc định:

```powershell
python src/main.py --model MASKED_GLORIA_CF --dataset book
```

Nếu thêm CLI args:

```powershell
python src/main.py `
  --model MASKED_GLORIA_CF `
  --dataset book `
  --cf_lambda 0.1 `
  --cf_warmup_ratio 0.1 `
  --cf_user_ratio 0.1 `
  --cf_k 20 `
  --cf_boundary_width 5 `
  --cf_boundary_q 3 `
  --cf_temperature 1.0
```

---

## 22. Thứ tự implement khuyến nghị

1. Copy `masked_gloria.py` -> `masked_gloria_cf.py` và đổi class.
2. Thêm config CF + epoch hooks.
3. Build `user_to_edge_ids` và `user_seen_items` từ `forward_edge_users/items`.
4. Tách `loss_rec` ra helper riêng.
5. Implement probe ranking với `mask.detach()` và `torch.no_grad()`.
6. Implement fragile detection bằng rank.
7. Implement CF forward không `no_grad`, dùng detached mask.
8. Implement pairwise boundary loss.
9. Restore/cache `result_embed` an toàn.
10. Thêm logging trong `post_epoch_processing()`.
11. Smoke test với `cf_lambda=0`.
12. Test gradient isolation.
13. Chạy experiment nhỏ để kiểm tra tốc độ và số fragile sample.

---

## 23. Ghi chú cuối

Plan này ưu tiên implementation ít xâm lấn: tạo model mới và giữ nguyên `MASKED_GLORIA`. Điểm quan trọng nhất của Exp A là `mask_cf = mask.detach()`: nếu quên detach, experiment sẽ biến thành mask supervision và không còn đúng mục tiêu representation robustness.
