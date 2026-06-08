#!/bin/bash
# =============================================================================
# Batería completa (Aguilar2026Features) como UN SOLO job de 64 cores en
# gpuserver02 (partición lsc).
#
# Por qué un job único y no el array (scripts/run_array_slurm.sh, ahora obsoleto):
# run_experiment.py funde las grillas de identificación de los TRES datasets en
# un único pool plano de joblib (cross_validate_tasks). Los folds de las cohortes
# chicas (BIDMC, PTT) rellenan la cola de la grande (MIMIC) en vez de quedar
# ociosos, así que hay una sola cola de scheduling para toda la batería en lugar
# de tres procesos de 21 cores que se estrangulan mutuamente. El array repartía
# 3x21 cores fijos a datasets desiguales (MIMIC estrangulado, BIDMC/PTT ociosos)
# y, peor aún, sus tres procesos separados NO pueden compartir el pool.
#
# Se pide --gres=gpu:1 para el path CuPy de _reconstruct_chunk_gpu
# (non_invertibility.py): QR por lotes + GEMMs en A100 vs N QR secuenciales CPU.
# La carga principal (CV sklearn) no usa GPU; el overhead de reservar la tarjeta
# es mínimo frente al tiempo de reconstrucción con la cohorte completa.
#
# Envío:
#     sbatch scripts/run_slurm.sh
# Monitoreo:
#     squeue -u "$USER"
#     tail -f logs/run_<jobid>.out
# Resultados (incluida la tabla comparativa global shared/dataset_comparison_*):
#     results/run_<jobid>/
# =============================================================================

#SBATCH --job-name=aguilar                  # Nombre del job
#SBATCH --partition=lsc                      # Cola/partición (única del nodo)
#SBATCH --nodes=1                            # Un nodo (gpuserver02)
#SBATCH --ntasks=1                           # Un proceso: el pool plano vive dentro
#SBATCH --cpus-per-task=64                   # 64 <= 70 cores; deja margen al nodo compartido
#SBATCH --mem=160G                           # Holgado: 3 cohortes chicas en RAM <= 160 << 411 GB
#SBATCH --gres=gpu:1                         # 1 A100 para _reconstruct_chunk_gpu
#SBATCH --time=2-00:00:00                    # Límite generoso (la cola es INFINITE)
#SBATCH --output=logs/run_%j.out             # stdout (%j = job id)
#SBATCH --error=logs/run_%j.err              # stderr

set -euo pipefail

# --- Ubicación del proyecto -------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"
mkdir -p logs

# Árbol de salida propio del job; main() escribe también shared/ con la tabla
# comparativa global, así que ya no hace falta el job de agregación.
JOB_ID="${SLURM_JOB_ID:-manual}"
OUT_DIR="$PROJECT_DIR/results/run_${JOB_ID}"

# --- Núcleos asignados ------------------------------------------------------
NCORES="${SLURM_CPUS_PER_TASK:-64}"

# --- Reproducibilidad bit-a-bit ---------------------------------------------
export PYTHONHASHSEED=42

# --- Control de paralelismo -------------------------------------------------
# Fijado a los cores del cgroup (NO -1): joblib con -1 vería os.cpu_count()=70 y
# sobre-suscribiría el nodo compartido más allá de los 64 cores asignados.
export AGUILAR_FEATURES_CV_N_JOBS="$NCORES"   # pool plano de CV (pipeline.py)
export AGUILAR_FEATURES_N_JOBS="$NCORES"      # carga/limpieza por filas (batch_utils.py)

# Cada worker de joblib debe ser mono-hilo en BLAS para no sobre-suscribir.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# Memmap de joblib en disco local rápido: con N workers compartiendo las matrices
# de plantillas, /tmp en red mata el rendimiento.
export JOBLIB_TEMP_FOLDER="${SLURM_TMPDIR:-/dev/shm}"

# Barras de progreso (tqdm) desactivadas: en un job batch el stdout va a un
# archivo, donde los retornos de carro de tqdm lo inundarían. Los logs de "Step:"
# de -v ya narran el avance. Quitar esta línea para verlas en una sesión interactiva.
export AGUILAR_FEATURES_NO_PROGRESS=1

echo "==================================================================="
echo "Nodo:            $(hostname)"
echo "Job:             ${JOB_ID}"
echo "Cores asignados: $NCORES"
echo "Salida:          $OUT_DIR"
echo "Memmap temp:     $JOBLIB_TEMP_FOLDER"
echo "Inicio:          $(date)"
echo "==================================================================="

# --- GPU visible para CuPy ---------------------------------------------------
# Restringir a la tarjeta asignada por SLURM; si no se exporta, CuPy ve todas.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# --- Entorno con uv (idempotente) -------------------------------------------
uv sync --frozen
# CuPy no está en el lockfile (el wheel depende del driver CUDA del nodo).
# uv pip install es idempotente y no toca el lockfile de los deps base.
uv pip install cupy-cuda12x

# --- Batería completa: los 3 datasets en un pool compartido -----------------
uv run python scripts/run_experiment.py --datasets all --all --tune \
    --output-dir "$OUT_DIR" -v

echo "==================================================================="
echo "Fin: $(date)"
echo "Resultados en:     $OUT_DIR/"
echo "Tabla global:      $OUT_DIR/shared/dataset_comparison_*.csv"
echo "==================================================================="
