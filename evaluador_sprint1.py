# ╔══════════════════════════════════════════════════════════════════╗
# ║         EVALUADOR AUTOMÁTICO — SPRINT #1 DEEP LEARNING          ║
# ║         Universidad Autónoma de Occidente — Maestría IA         ║
# ║         Dr. Carlos Andrés Ferro Sánchez · 2026                  ║
# ╚══════════════════════════════════════════════════════════════════╝

import os, sys, time
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (f1_score, classification_report,
                             confusion_matrix, precision_score, recall_score)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════
# SECCIÓN 0 — CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════

NOMBRE_ESTUDIANTE = "Isabella - Martin - Luz Angela"
RUTA_MODELO       = "best_model.pt"
RUTA_DATOS        = "./blind_test"
IMG_SIZE          = 128

# ══════════════════════════════════════════════════════════════════
# SECCIÓN 1 — ARQUITECTURA  (TomatoCNN_VGG — idéntica al notebook)
# ══════════════════════════════════════════════════════════════════

class ConvBlock(nn.Module):
    """
    Bloque VGG: [Conv → BN → ReLU] × n_convs → MaxPool2d → Dropout2d
    DEBE ser idéntico al usado en entrenamiento.
    """
    def __init__(self, in_ch, out_ch, n_convs=2, dropout2d=0.1):
        super().__init__()
        layers = []
        for i in range(n_convs):
            c_in = in_ch if i == 0 else out_ch
            layers += [
                nn.Conv2d(c_in, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
        layers += [nn.MaxPool2d(2), nn.Dropout2d(dropout2d)]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class MiCNN(nn.Module):
    """
    TomatoCNN_VGG  — 5 bloques ConvBlock + AdaptiveAvgPool + 3 FC.
    Nombre MiCNN para no cambiar el resto del evaluador del docente.
    """
    def __init__(self, num_classes=4):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(3,   32,  n_convs=2, dropout2d=0.05),   # 128→64
            ConvBlock(32,  64,  n_convs=2, dropout2d=0.05),   # 64→32
            ConvBlock(64,  128, n_convs=3, dropout2d=0.10),   # 32→16
            ConvBlock(128, 256, n_convs=3, dropout2d=0.10),   # 16→8
            ConvBlock(256, 256, n_convs=2, dropout2d=0.15),   # 8→4
        )

        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(1024, 256),          nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


# ══════════════════════════════════════════════════════════════════
# SECCIÓN 2 — EVALUACIÓN (NO MODIFICAR)
# ══════════════════════════════════════════════════════════════════

CLASES_ESPERADAS = [
    "Tomato_Early_Blight",
    "Tomato_Healthy",
    "Tomato_Late_Blight",
    "Tomato_Yellow_Leaf_Curl_Virus",
]

COLORES_CLASE = {
    "Tomato_Healthy":               "#4CAF50",
    "Tomato_Early_Blight":          "#FF9800",
    "Tomato_Late_Blight":           "#F44336",
    "Tomato_Yellow_Leaf_Curl_Virus":"#9C27B0",
}

def get_nivel(f1):
    if f1 >= 0.93: return ("EXCELENCIA",       "TOP",    "#1B5E20", "#E8F5E9", 5.0)
    if f1 >= 0.86: return ("OBJETIVO SPRINT",  "MID",    "#E65100", "#FFF8E1", 4.0)
    if f1 >= 0.80: return ("MINIMO ACEPTABLE", "LOW",    "#1565C0", "#E3F2FD", 2.5)
    return             ("POR DEBAJO",          "FAIL",   "#B71C1C", "#FFEBEE", 1.0)

def separador(c="═", n=60): print(c * n)

def evaluar():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    separador()
    print("  EVALUADOR SPRINT #1 — DEEP LEARNING")
    print(f"  Estudiante : {NOMBRE_ESTUDIANTE}")
    print(f"  Device     : {DEVICE}")
    separador()

    # ── Validar archivos ────────────────────────────────────────
    errores = []
    if not os.path.exists(RUTA_MODELO):
        errores.append(f"  [ERROR] Modelo no encontrado: {RUTA_MODELO}")
    if not os.path.exists(RUTA_DATOS):
        errores.append(f"  [ERROR] Carpeta de datos no encontrada: {RUTA_DATOS}")
    if errores:
        for e in errores: print(e)
        print("\n  Revisa las rutas en la Seccion 0 y vuelve a ejecutar.")
        sys.exit(1)

    # ── Verificar clases ────────────────────────────────────────
    clases_disco = sorted(os.listdir(RUTA_DATOS))
    clases_disco = [c for c in clases_disco if os.path.isdir(os.path.join(RUTA_DATOS, c))]
    print(f"\n  Clases encontradas ({len(clases_disco)}):")
    for c in clases_disco:
        n_imgs = len(os.listdir(os.path.join(RUTA_DATOS, c)))
        print(f"    {c:<45} {n_imgs:>5} imagenes")

    # ── Dataset ─────────────────────────────────────────────────
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    ds     = datasets.ImageFolder(RUTA_DATOS, transform=tf)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    clases = ds.classes
    print(f"\n  Total imagenes a evaluar: {len(ds)}")

    # ── Cargar modelo ───────────────────────────────────────────
    print(f"\n  Cargando modelo desde: {RUTA_MODELO}")
    try:
        model = MiCNN(num_classes=len(clases)).to(DEVICE)

        # ── FIX: nuestro checkpoint es un dict con 'model_state_dict' ──
        raw = torch.load(RUTA_MODELO, map_location=DEVICE)
        if isinstance(raw, dict) and "model_state_dict" in raw:
            state = raw["model_state_dict"]
            print(f"  Checkpoint dict detectado  (época {raw.get('epoch','?')}, "
                  f"val_f1={raw.get('val_f1', '?')})")
        else:
            state = raw   # guardado directo de state_dict (fallback)

        model.load_state_dict(state)
        model.eval()

        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Parametros entrenables: {params:,}")

    except Exception as e:
        print(f"\n  [ERROR] No se pudo cargar el modelo: {e}")
        print("  Asegurate de que la arquitectura en Seccion 1 sea identica a la usada en entrenamiento.")
        sys.exit(1)

    # ── Inferencia ──────────────────────────────────────────────
    print("\n  Evaluando... ", end="", flush=True)
    t0 = time.time()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs   = imgs.to(DEVICE)
            logits = model(imgs)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            preds  = logits.argmax(1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    elapsed = time.time() - t0
    print(f"listo en {elapsed:.1f}s")

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    # ── Métricas ────────────────────────────────────────────────
    f1_macro   = f1_score(y_true, y_pred, average="macro")
    f1_clase   = f1_score(y_true, y_pred, average=None)
    prec_macro = precision_score(y_true, y_pred, average="macro")
    rec_macro  = recall_score(y_true, y_pred, average="macro")
    acc        = np.mean(y_true == y_pred)
    cm         = confusion_matrix(y_true, y_pred)
    report     = classification_report(y_true, y_pred,
                                       target_names=clases, digits=4)

    nivel_txt, nivel_key, color_nivel, bg_nivel, nota = get_nivel(f1_macro)

    # ── Consola ─────────────────────────────────────────────────
    separador()
    print(f"\n  F1-Score Macro  : {f1_macro:.4f}")
    print(f"  Accuracy        : {acc:.4f}")
    print(f"  Precision Macro : {prec_macro:.4f}")
    print(f"  Recall Macro    : {rec_macro:.4f}")
    print(f"\n  F1 por clase:")
    for cls, f1 in zip(clases, f1_clase):
        bar = "█" * int(f1 * 20)
        print(f"    {cls:<45} {f1:.4f}  {bar}")
    separador()
    print(f"\n  NIVEL ALCANZADO : {nivel_txt}")
    print(f"  NOTA MAXIMA     : {nota} / 5.0")
    separador()
    print(f"\n{report}")

    # ── FIGURA ──────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 13), facecolor="#FAFAFA")
    fig.suptitle(
        f"Sprint #1 — Clasificación de Enfermedades en Tomate\n{NOMBRE_ESTUDIANTE}",
        fontsize=15, fontweight="bold", color="#1A1A2E", y=0.98
    )

    gs = gridspec.GridSpec(3, 3, figure=fig,
                           hspace=0.45, wspace=0.35,
                           top=0.92, bottom=0.06, left=0.07, right=0.97)

    # Panel 1: Banner resultado
    ax_banner = fig.add_subplot(gs[0, :])
    ax_banner.set_facecolor(bg_nivel)
    ax_banner.set_xlim(0, 1); ax_banner.set_ylim(0, 1)
    ax_banner.axis("off")
    barra_w = min(f1_macro, 1.0)
    ax_banner.add_patch(mpatches.FancyBboxPatch(
        (0.01, 0.08), 0.98, 0.28, boxstyle="round,pad=0.01",
        facecolor="#E0E0E0", edgecolor="none"))
    ax_banner.add_patch(mpatches.FancyBboxPatch(
        (0.01, 0.08), 0.98 * barra_w, 0.28, boxstyle="round,pad=0.01",
        facecolor=color_nivel, edgecolor="none", alpha=0.85))
    ax_banner.text(0.50, 0.78, f"F1-Score Macro: {f1_macro:.4f}",
                   ha="center", va="center", fontsize=28, fontweight="bold",
                   color=color_nivel, transform=ax_banner.transAxes)
    ax_banner.text(0.50, 0.55, f"{nivel_txt}   |   Nota máxima: {nota} / 5.0",
                   ha="center", va="center", fontsize=16, color="#333333",
                   transform=ax_banner.transAxes)
    for umbral, lbl in [(0.80, "0.80"), (0.86, "0.86"), (0.93, "0.93")]:
        x = 0.01 + 0.98 * umbral
        ax_banner.axvline(x=x, ymin=0.05, ymax=0.42,
                          color="#555555", lw=1.5, ls="--", alpha=0.6,
                          transform=ax_banner.transAxes)
        ax_banner.text(x, 0.43, lbl, ha="center", fontsize=9,
                       color="#555555", transform=ax_banner.transAxes)

    # Panel 2: Matriz de confusión
    ax_cm = fig.add_subplot(gs[1:, 0:2])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    labels_short = [c.replace("Tomato_", "").replace("_", "\n") for c in clases]
    sns.heatmap(cm_norm, annot=False, cmap="Blues",
                xticklabels=labels_short, yticklabels=labels_short,
                linewidths=0.5, linecolor="#CCCCCC",
                ax=ax_cm, cbar_kws={"shrink": 0.7})
    for i in range(len(clases)):
        for j in range(len(clases)):
            val_n = cm_norm[i, j]
            val_a = cm[i, j]
            color_txt = "white" if val_n > 0.55 else "black"
            ax_cm.text(j + 0.5, i + 0.45, f"{val_n:.1%}",
                       ha="center", va="center", fontsize=11,
                       fontweight="bold", color=color_txt)
            ax_cm.text(j + 0.5, i + 0.65, f"({val_a})",
                       ha="center", va="center", fontsize=8,
                       color=color_txt, alpha=0.8)
    ax_cm.set_title("Matriz de Confusión (% por fila real)", fontsize=12,
                    fontweight="bold", pad=10)
    ax_cm.set_ylabel("Clase Real", fontsize=10)
    ax_cm.set_xlabel("Clase Predicha", fontsize=10)
    ax_cm.tick_params(axis="both", labelsize=9)

    # Panel 3: F1 por clase
    ax_f1 = fig.add_subplot(gs[1, 2])
    colores_barra = [COLORES_CLASE.get(c, "#607D8B") for c in clases]
    bars = ax_f1.barh(
        [c.replace("Tomato_", "").replace("_", " ") for c in clases],
        f1_clase, color=colores_barra, edgecolor="white", height=0.6
    )
    ax_f1.axvline(x=f1_macro, color="#C8102E", lw=2, ls="--",
                  label=f"Macro={f1_macro:.3f}")
    ax_f1.set_xlim(0, 1.05)
    for bar, val in zip(bars, f1_clase):
        ax_f1.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                   f"{val:.3f}", va="center", fontsize=9, fontweight="bold")
    ax_f1.set_title("F1-Score por Clase", fontsize=11, fontweight="bold")
    ax_f1.set_xlabel("F1-Score", fontsize=9)
    ax_f1.legend(fontsize=8)
    ax_f1.tick_params(labelsize=8)
    ax_f1.grid(axis="x", alpha=0.3)
    ax_f1.set_facecolor("#FAFAFA")

    # Panel 4: Resumen métricas
    ax_met = fig.add_subplot(gs[2, 2])
    metricas_nombres = ["Accuracy", "Precision\nMacro", "Recall\nMacro", "F1\nMacro"]
    metricas_vals    = [acc, prec_macro, rec_macro, f1_macro]
    colores_met      = ["#1565C0", "#2E7D32", "#E65100", "#C8102E"]
    bars2 = ax_met.bar(metricas_nombres, metricas_vals,
                       color=colores_met, edgecolor="white",
                       width=0.55, alpha=0.88)
    ax_met.set_ylim(0, 1.12)
    ax_met.axhline(y=0.80, color="#888888", lw=1.2, ls="--", alpha=0.7)
    ax_met.axhline(y=0.86, color="#FF9800", lw=1.2, ls="--", alpha=0.7)
    ax_met.axhline(y=0.93, color="#4CAF50", lw=1.2, ls="--", alpha=0.7)
    for bar, val in zip(bars2, metricas_vals):
        ax_met.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                    f"{val:.3f}", ha="center", fontsize=10, fontweight="bold")
    ax_met.text(1.03, 0.80, "Min", transform=ax_met.get_xaxis_transform(),
                fontsize=7, color="#888888", va="center")
    ax_met.text(1.03, 0.86, "Obj", transform=ax_met.get_xaxis_transform(),
                fontsize=7, color="#FF9800", va="center")
    ax_met.text(1.03, 0.93, "Top", transform=ax_met.get_xaxis_transform(),
                fontsize=7, color="#4CAF50", va="center")
    ax_met.set_title("Resumen de Métricas", fontsize=11, fontweight="bold")
    ax_met.set_ylabel("Score", fontsize=9)
    ax_met.tick_params(labelsize=8)
    ax_met.set_facecolor("#FAFAFA")
    ax_met.grid(axis="y", alpha=0.3)

    # Footer
    fig.text(0.5, 0.01,
             f"Sprint #1 · UAO · Maestría IA · {NOMBRE_ESTUDIANTE} · "
             f"Imágenes evaluadas: {len(ds)} · Tiempo: {elapsed:.1f}s",
             ha="center", fontsize=8, color="#888888")

    fname = f"reporte_sprint1_{NOMBRE_ESTUDIANTE.replace(' ','_')}.png"
    plt.savefig(fname, dpi=160, bbox_inches="tight", facecolor="#FAFAFA")
    plt.show()

    separador("─")
    print(f"  Imagen guardada: {fname}")
    separador("═")
    print(f"\n  RESULTADO FINAL")
    print(f"  ┌──────────────────────────────────────────┐")
    print(f"  │  Estudiante : {NOMBRE_ESTUDIANTE:<27}│")
    print(f"  │  F1 Macro   : {f1_macro:<27.4f}│")
    print(f"  │  Nivel      : {nivel_txt:<27}│")
    print(f"  │  Nota max   : {nota:<27}│")
    print(f"  └──────────────────────────────────────────┘\n")

    if nivel_key == "FAIL":
        print("  Sugerencia: Revisa el manejo del desbalance de clases,")
        print("  aumenta epocas o ajusta el learning rate.")
    elif nivel_key == "LOW":
        print("  Bien encaminado. Prueba: mas data augmentation,")
        print("  WeightedRandomSampler, o ajuste fino del LR.")
    elif nivel_key == "MID":
        print("  Excelente trabajo. Para llegar al Top: profundiza la")
        print("  arquitectura, usa label smoothing o CosineAnnealingLR.")
    else:
        print("  Resultado de excelencia. Modelo solido y bien entrenado.")
    print()

if __name__ == "__main__":
    evaluar()