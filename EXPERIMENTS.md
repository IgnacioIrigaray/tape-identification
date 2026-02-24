# Experimentos — Tape Parameter Identification

## Arquitectura base

| Componente | Configuracion |
|---|---|
| **Encoder** | SpectralEncoder (MobileNetV2 `width_mult=2` o EfficientNet-B2), `embed_dim=1024`, STFT log-dB con floor -80 dB |
| **Controller** | MLP 1024-256-256-1, LeakyReLU, Dropout(0.1), sin activacion de salida |
| **Activacion** | Sigmoid en el trainer, prediccion en dB: `pred_dB = s(z) * range + min_param` |
| **Loss** | MSE en espacio dB |
| **Audio** | sr=22050, 65536 samples (~3s), peak-normalizado |
| **Entrenamiento** | Adam lr=1e-3, weight_decay=1e-5, ReduceLROnPlateau (factor=0.5, patience=5), grad_clip=1.0, patience=100 |
| **Hardware** | RTX 3060 12GB VRAM, 25GB RAM |

### Datasets

| Dataset | Contenido | Formato | Archivos |
|---|---|---|---|
| **Jamendo** | Musica variada (pop, rock, electronica, clasica) | mp3 | ~1850 |
| **GuitarSet** | Guitarra acustica solista, grabaciones limpias | wav | ~360 |

Split: 80% train / 10% val / 10% test (seed=42, deterministico).

---

## Experimentos exploratorios (historico)

| Exp | Degradacion | Tarea | Resultado | Notas |
|---|---|---|---|---|
| 1 | HardClipping | Clasif 3 clases | 90.5% acc | Primer baseline funcional |
| 2 | HardClipping | Clasif 10 clases | 30% acc | Demasiadas clases, estancado |
| 3 | Wow/Flutter | Clasif 3 clases (rate) | ~88.5% acc | Sin modelo guardado |
| 4 | JA + WF | Clasif dual 3x3 | -- | Paso intermedio |
| 5 | JA + WF | Regresion triple | R2~0 | Problemas de OOM, nunca entreno bien |

---

## Fase 1 -- Baselines individuales

Un modelo por degradacion. Cada uno estima un solo parametro continuo (regresion).
Se entrena con cada dataset para evaluar el impacto del dominio de audio.

### Jamendo

| ID | Degradacion | Parametro | Rango | Config | Exp name | MAE | RMSE | R2 |
|---|---|---|---|---|---|---|---|---|
| 1a | JA hysteresis | drive | [1, 10] | `configs/ja.yaml` | `exp_ja_jamendo` | | | |
| 1b | Tanh saturation | gain | [1, 10] | `configs/tanh.yaml` | `exp_tanh_jamendo` | | | |
| 1c | Hard clipping | gain | [1, 4] | `configs/hard_clipping.yaml` | `exp_hc_jamendo` | | | |
| 1d | Wow/Flutter | depth | [0.1, 0.8] | `configs/wow_flutter.yaml` | `exp_wf_jamendo` | | | |
| 1e | Tape noise | SNR (dB) | [10, 30] | `configs/tape_noise.yaml` | `exp_noise_jamendo` | | | |

### GuitarSet

| ID | Degradacion | Parametro | Rango | Config | Exp name | MAE | RMSE | R2 |
|---|---|---|---|---|---|---|---|---|
| 2a | JA hysteresis | drive | [1, 10] | `configs/ja.yaml` | `exp_ja_guitarset` | | | |
| 2b | Tanh saturation | gain | [1, 10] | `configs/tanh.yaml` | `exp_tanh_guitarset` | | | |
| 2c | Hard clipping | gain | [1, 4] | `configs/hard_clipping.yaml` | `exp_hc_guitarset` | | | |
| 2d | Wow/Flutter | depth | [0.1, 0.8] | `configs/wow_flutter.yaml` | `exp_wf_guitarset` | | | |
| 2e | Tape noise | SNR (dB) | [10, 30] | `configs/tape_noise.yaml` | `exp_noise_guitarset` | | | |

**Para entrenar con GuitarSet**: cambiar `audio_dir` y `ext` en el config:
```yaml
audio_dir: /mnt/data/working_datasets/guitarset
ext: wav
```

---

## Fase 2 -- Modelos combinados

Degradaciones aplicadas en cadena. El modelo estima todos los parametros simultaneamente.

| ID | Cadena | Parametros | Heads | Requiere codigo |
|---|---|---|---|---|
| 3a | JA -> tape_noise | drive + SNR | 2 | `_apply_ja_noise()` en dataset.py |
| 3b | JA -> WF -> tape_noise | drive + depth + rate + SNR | 4 | Extender triple a quad |
| 3c | JA -> WF (existente) | drive + depth + rate | 3 | Ya implementado (`ja_wf`) |

### Resultados combinados (Jamendo)

| ID | Cadena | MAE drive | MAE depth | MAE rate | MAE SNR | R2 global |
|---|---|---|---|---|---|---|
| 3a | JA -> noise | | | -- | | |
| 3b | JA -> WF -> noise | | | | | |
| 3c | JA -> WF | | | | -- | |

---

## Fase 3 -- Evaluacion cross-domain

Entrenar en un dataset, evaluar en el otro. Mide la capacidad de generalizacion del modelo.

| ID | Train | Eval | Degradacion | MAE (train domain) | MAE (cross domain) | Delta |
|---|---|---|---|---|---|---|
| 4a | Jamendo | GuitarSet | JA | | | |
| 4b | GuitarSet | Jamendo | JA | | | |
| 4c | Jamendo | GuitarSet | tape_noise | | | |
| 4d | GuitarSet | Jamendo | tape_noise | | | |
| 4e | Jamendo | GuitarSet | wow_flutter | | | |
| 4f | GuitarSet | Jamendo | wow_flutter | | | |

---

## Comandos de referencia

```bash
# Entrenar
python scripts/train.py --config configs/ja.yaml --name exp_ja_jamendo

# Evaluar
python scripts/evaluate.py --name exp_ja_jamendo --plot --max_files 200 --device cpu

# Cross-domain: entrenar en Jamendo, evaluar manualmente con GuitarSet
# (cambiar audio_dir en el config de evaluacion)
```

---

## Notas

- **Fase 1** no requiere cambios de codigo, solo configs
- **Fase 2** requiere extender dataset.py (nuevos models combinados) y controller.py (mas heads)
- **Fase 3** reutiliza modelos de Fase 1, solo cambia el dataset de evaluacion
- Los entrenamientos se hacen de a uno (1 GPU, 12GB VRAM)
- Todos los experimentos usan la misma seed (42) para el split de datos
