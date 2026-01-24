# TensorBoard Logging

El sistema de entrenamiento incluye logging completo con TensorBoard para monitorear la evolución del entrenamiento y los parámetros.

## Métricas Registradas

### Durante Entrenamiento (por step)
- **train/loss_step**: Loss de entrenamiento en cada batch
- **train/depth_mean**: Media de los parámetros de depth predichos
- **train/depth_std**: Desviación estándar de los parámetros predichos
- **train/depth_min**: Valor mínimo de depth predicho
- **train/depth_max**: Valor máximo de depth predicho
- **train/learning_rate**: Learning rate actual

### Durante Validación (por época)
- **val/loss**: Loss de validación promedio
- **val/depth_mean**: Media de predicciones en validación
- **val/depth_std**: Desviación estándar en validación
- **val/depth_min**: Mínimo en validación
- **val/depth_max**: Máximo en validación
- **val/depth_distribution**: Histograma de distribución de predicciones

### Por Época
- **train/loss_epoch**: Loss de entrenamiento promedio por época
- **loss_comparison**: Comparación train vs val loss

## Iniciar TensorBoard

Una vez iniciado el entrenamiento, los logs se guardan en `outputs/logs/`. Para visualizarlos:

```bash
cd tape-identification
tensorboard --logdir=outputs/logs
```

Luego abre tu navegador en: http://localhost:6006

## Visualizaciones Recomendadas

### 1. Monitoring de Loss
En la pestaña **SCALARS**:
- Compara `train/loss_epoch` vs `val/loss` para detectar overfitting
- Observa `loss_comparison` para ver ambas curvas juntas

### 2. Evolución de Parámetros
En la pestaña **SCALARS**:
- `train/depth_mean` y `val/depth_mean`: Verifica que las predicciones converjan
- `train/depth_std`: Desviación estándar debe disminuir conforme el modelo aprende
- `train/depth_min` y `train/depth_max`: Rango de predicciones (debe estar entre 1 y 10)

### 3. Distribución de Predicciones
En la pestaña **HISTOGRAMS**:
- `val/depth_distribution`: Histograma de predicciones de depth
- Idealmente debería cubrir uniformemente el rango [1, 10]

### 4. Learning Rate
En la pestaña **SCALARS**:
- `train/learning_rate`: Monitorear si usas schedulers

## Estructura de Archivos

```
outputs/
├── logs/                    # TensorBoard logs
│   └── events.out.tfevents.*
└── checkpoints/            # Model checkpoints
    ├── best_model.pt
    └── checkpoint_epoch*.pt
```

## Tips

1. **Múltiples Runs**: Cada ejecución crea un subdirectorio con timestamp en `logs/`, permitiendo comparar diferentes experimentos.

2. **Smoothing**: En TensorBoard, ajusta el smoothing slider para suavizar curvas ruidosas.

3. **Comparación**: Para comparar múltiples runs:
   ```bash
   tensorboard --logdir=outputs/logs --reload_multifile=true
   ```

4. **Remote Access**: Si entrenas en un servidor remoto:
   ```bash
   # En el servidor
   tensorboard --logdir=outputs/logs --bind_all

   # Desde tu máquina local
   ssh -L 6006:localhost:6006 user@server
   ```

## Debugging

Si encuentras problemas:

1. **No aparecen métricas**: Verifica que el directorio `outputs/logs/` existe y contiene archivos `events.out.tfevents.*`

2. **TensorBoard no inicia**: Verifica que tensorboard esté instalado:
   ```bash
   pip install tensorboard
   ```

3. **Puerto ocupado**: Usa otro puerto:
   ```bash
   tensorboard --logdir=outputs/logs --port=6007
   ```
