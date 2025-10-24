Model này là một **Transformer-based hybrid architecture** được thiết kế chuyên biệt cho bài toán **drug–side effect prediction** (dự đoán tác dụng phụ của thuốc).
Nó kết hợp giữa **representation learning (Transformer)** và **interaction modeling (Outer Product + CNN)** để nắm bắt mối quan hệ phức tạp giữa **thuốc** và **tác dụng phụ**.

---

## 🧩 1. **Overall Pipeline**

```
Drug (SMILES/token seq) ─┐
                         ├─> Drug Encoder ─┐
                         │                 │
                         │                 │
Side Effect (term seq) ──┘                 ├─> Interaction (outer product + CNN) → MLP → Score
                                           │
                                           └─> (Optional) Cross Attention
```

---

## 🔹 2. **Embedding Layers**

### `self.emb_drug` & `self.emb_side`

Hai lớp embedding này chuyển đổi chỉ số token (thuốc hoặc tác dụng phụ) thành vector liên tục `hidden_size` chiều, đồng thời thêm positional embedding và dropout.

* Input: `[batch, seq_len]`
* Output: `[batch, seq_len, hidden_size]`

> Mỗi “thuốc” và “tác dụng phụ” được biểu diễn như chuỗi token (vd: substructures hoặc words), sau đó được ánh xạ vào không gian vector.

---

## 🔹 3. **Transformer Encoders**

### `self.encoder_drug` và `self.encoder_side`

Hai khối encoder độc lập — mỗi bên một Transformer nhiều tầng (`Encoder_MultipleLayers`).

Chúng học **biểu diễn ngữ cảnh nội tại**:

* Trong chuỗi thuốc: mô hình học mối quan hệ giữa các substructures.
* Trong chuỗi side effect: mô hình học ngữ nghĩa giữa các thành phần của mô tả tác dụng phụ.

Cấu trúc mỗi encoder layer gồm:

* Multi-head self-attention
* Feed-forward network
* Residual + LayerNorm
* (Tùy chọn: FlashAttention / SDPA / gradient checkpointing)

> Mục tiêu: học được embedding “cấp câu” (contextualized embeddings).

---

## 🔹 4. **Optional Cross-Attention Encoder**

Nếu `config.use_cross_attention=True`, mô hình thêm 1 tầng Transformer để **cho phép hai chuỗi tương tác trực tiếp ở mức attention**.

Cụ thể:

```python
combined = torch.cat([drug_encoded, se_encoded], dim=1)
combined = cross_attention_encoder(combined)
```

→ giúp mô hình hiểu được **mối quan hệ giữa từng token của thuốc và từng token của side effect**, thay vì xử lý tách biệt.

---

## 🔹 5. **Interaction Layer (Outer Product + CNN)**

Đây là phần cốt lõi để mô hình hóa **tương tác hai chiều giữa thuốc và tác dụng phụ**.

### ✳ Cơ chế:

1. Mở rộng chiều để broadcast:

   ```python
   drug_aug = [batch, drug_len, 1, hidden]
   se_aug   = [batch, 1, se_len, hidden]
   ```

2. Outer product:

   ```python
   interaction = drug_aug * se_aug
   # => [batch, drug_len, se_len, hidden]
   ```

   → tương tự như tạo một “ma trận tương tác” giữa mỗi token thuốc và mỗi token side effect.

3. Hoán vị cho CNN:

   ```python
   interaction = interaction.permute(0, 3, 1, 2)
   # => [batch, hidden, drug_len, se_len]
   ```

4. Giảm chiều hidden bằng tổng:

   ```python
   interaction = torch.sum(interaction, dim=1, keepdim=True)
   # => [batch, 1, drug_len, se_len]
   ```

5. Áp dụng CNN để trích xuất đặc trưng cục bộ:

   ```python
   interaction_features = self.interaction_cnn(interaction)
   ```

   → CNN giúp học các **pattern cục bộ** giữa subsequences (vd: motif liên quan đến tác dụng phụ cụ thể).

6. Flatten để đưa vào decoder.

---

## 🔹 6. **Decoder (MLP)**

Phần cuối là một mạng **Multi-Layer Perceptron** gồm nhiều tầng linear + ReLU (+ BatchNorm nếu bật).

Nhiệm vụ:

* Nhận đầu vào là vector flattened từ tầng CNN.
* Dự đoán xác suất (hoặc score liên tục) của “thuốc có gây tác dụng phụ này hay không”.

Output:

```python
score = self.decoder(interaction_flat)
# [batch, 1]
```

---

## 🔹 7. **Training Objective**

Tùy dataset, loss thường là:

* **Binary cross-entropy (BCE)**: nếu bài toán nhị phân (có/không tác dụng phụ)
* **Regression loss (MSE)**: nếu label là mức độ nghiêm trọng

---

## 🔹 8. **Attention Mask**

Hàm `_create_attention_mask` biến mask `[batch, seq_len]` thành `[batch, 1, 1, seq_len]` với:

* 1 → token hợp lệ
* 0 → padding (bị phạt `-10000` trong attention logits)

---

## 🔹 9. **Inference & Feature Extraction**

`get_embeddings()` cho phép trích xuất embedding đã encode của thuốc & side effect mà không cần tính dự đoán.
→ hữu ích khi dùng làm feature cho downstream task (vd clustering, visualization).

---

## 🔹 10. **Parameter Breakdown**

`count_parameters()` thống kê:

* Tổng số tham số
* Tham số trainable
* Số tham số riêng của từng module (encoder, decoder)

---

## ⚙️ Tóm tắt dòng chảy dữ liệu

```
drug_seq --> Embedding --> Transformer --> drug_encoded ┐
                                                         ├─> Outer Product -> CNN -> Flatten -> MLP -> Score
side_seq --> Embedding --> Transformer --> se_encoded   ┘
```

Tùy `config`, có thể có thêm cross-attention giữa `drug_encoded` và `se_encoded`.

---

## 🧠 Ý nghĩa mô hình

* **Drug encoder** học representation từ SMILES hoặc substructure của thuốc.
* **Side effect encoder** học semantic embedding của tác dụng phụ.
* **Outer product + CNN** học cách mà các đặc trưng của thuốc “kết hợp” với nhau để gây ra một tác dụng phụ cụ thể.
* **MLP decoder** tổng hợp thông tin thành xác suất dự đoán cuối cùng.
