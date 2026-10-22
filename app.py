import os
import sys
import io
import shutil
import zipfile
import pathlib
from copy import deepcopy

import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import streamlit as st
from ultralytics import YOLO
from ultralytics.nn import tasks

# =====================================================================
# 1. DEFINICIÓN Y REGISTRO DE MÓDULOS PERSONALIZADOS DE GAF-NET
# =====================================================================

class GSConv(nn.Module):
    """Grouped Spatial Convolution del proyecto GAF-Net."""
    def __init__(self, c1, c2=None, k=3, s=1):
        super().__init__()
        if c2 is None:
            c2 = c1
        c_ = c2 // 2
        self.cv1 = nn.Conv2d(c1, c_, 1, s)
        self.dw = nn.Conv2d(c_, c_, k, 1, padding=k // 2, groups=c_)
        self.bn = nn.BatchNorm2d(c_)
        self.act = nn.SiLU()

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.act(self.bn(self.dw(x1)))
        y = torch.cat((x1, x2), dim=1)
        b, c, h, w = y.shape
        y = (
            y.reshape(b, 2, c // 2, h, w)
             .permute(0, 2, 1, 3, 4)
             .reshape(b, c, h, w)
        )
        return y


class FASFF(nn.Module):
    """Feature-Adaptive Spatial Feature Fusion del proyecto GAF-Net."""
    def __init__(self, c1, c2=256, level=0):
        super().__init__()
        if isinstance(c1, int):
            c1 = [c1, c1, c1]
        self.level = level
        self.c2 = c2
        self.reduce = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ci, c2, 1, 1, bias=False),
                nn.BatchNorm2d(c2),
                nn.SiLU(),
            )
            for ci in c1
        ])
        self.weight_layers = nn.ModuleList([
            nn.Conv2d(c2, 1, 1, 1, 0)
            for _ in range(3)
        ])

    def _resize_to(self, x, size):
        if x.shape[-2:] == size:
            return x
        if x.shape[-2] > size[0] or x.shape[-1] > size[1]:
            return F.adaptive_avg_pool2d(x, size)
        return F.interpolate(x, size=size, mode="nearest")

    def forward(self, x):
        assert isinstance(x, (list, tuple)) and len(x) == 3
        target_size = x[self.level].shape[-2:]
        feats = [
            self.reduce[i](self._resize_to(x[i], target_size))
            for i in range(3)
        ]
        weights = torch.stack([
            self.weight_layers[i](feats[i])
            for i in range(3)
        ], dim=0)
        weights = torch.softmax(weights, dim=0)
        fused = (
            weights[0] * feats[0]
            + weights[1] * feats[1]
            + weights[2] * feats[2]
        )
        return fused


def _get_out_channels(module):
    if isinstance(module, GSConv):
        return module.cv1.out_channels * 2
    if isinstance(module, nn.Sequential) and len(module) > 0:
        return _get_out_channels(module[-1])
    if hasattr(module, "cv2"):
        return _get_out_channels(module.cv2)
    if hasattr(module, "conv") and isinstance(module.conv, nn.Conv2d):
        return module.conv.out_channels
    for child in reversed(list(module.children())):
        result = _get_out_channels(child)
        if result is not None:
            return result
    return None


def patched_parse_model(d, ch, verbose=True):
    d2 = deepcopy(d)
    fasff_info = []
    gsconv_info = []

    for section in ("backbone", "head"):
        for i, layer in enumerate(d2.get(section, [])):
            f, n, m_name, args = layer
            if m_name == "FASFF":
                fasff_info.append((section, i, f, args, n))
                d2[section][i] = [-1, 1, "Conv", [args[0], 1, 1]]
            elif m_name == "GSConv":
                gsconv_info.append((section, i, f, args, n))
                d2[section][i] = [f, n, "Conv", args]

    model, save = tasks._original_parse_model(d2, ch, verbose)

    if not gsconv_info and not fasff_info:
        return model, save

    backbone_len = len(d.get("backbone", []))

    for section, idx, from_val, orig_args, n in gsconv_info:
        global_idx = idx if section == "backbone" else backbone_len + idx
        dummy = model[global_idx]
        conv_mod = dummy[0] if isinstance(dummy, nn.Sequential) else dummy

        c1_actual = conv_mod.conv.in_channels
        c2_actual = conv_mod.conv.out_channels
        k = orig_args[1] if len(orig_args) > 1 else 3
        s = orig_args[2] if len(orig_args) > 2 else 1
        n_actual = len(dummy) if isinstance(dummy, nn.Sequential) else 1

        if n_actual > 1:
            gs = nn.Sequential(*(
                GSConv(
                    c1_actual if j == 0 else c2_actual,
                    c2_actual,
                    k,
                    s if j == 0 else 1
                )
                for j in range(n_actual)
            ))
        else:
            gs = GSConv(c1_actual, c2_actual, k, s)

        gs.i = global_idx
        gs.f = from_val
        gs.type = "GSConv"
        gs.np = sum(p.numel() for p in gs.parameters())
        model[global_idx] = gs

    for section, idx, from_val, args, n in fasff_info:
        global_idx = idx if section == "backbone" else backbone_len + idx
        if isinstance(from_val, list):
            input_chs = [_get_out_channels(model[fi]) for fi in from_val]
        else:
            input_chs = [_get_out_channels(model[from_val])] * 3

        actual_c2 = _get_out_channels(model[global_idx])
        level = args[1] if len(args) > 1 else 0

        fasff = FASFF(input_chs, actual_c2, level)
        fasff.i = global_idx
        fasff.f = from_val
        fasff.type = "FASFF"
        fasff.np = sum(p.numel() for p in fasff.parameters())
        model[global_idx] = fasff

        if isinstance(from_val, list):
            for fidx in from_val:
                real = fidx if fidx >= 0 else global_idx + fidx
                if real not in save:
                    save.append(real)

    return model, save


def register_custom_modules():
    """Registra GSConv y FASFF en Ultralytics y en __main__."""
    tasks.GSConv = GSConv
    tasks.FASFF = FASFF

    import __main__
    __main__.GSConv = GSConv
    __main__.FASFF = FASFF
    sys.modules["__main__"].GSConv = GSConv
    sys.modules["__main__"].FASFF = FASFF

    try:
        torch.serialization.add_safe_globals([
            GSConv,
            FASFF,
            nn.Conv2d,
            nn.BatchNorm2d,
            nn.SiLU,
            nn.ModuleList,
            nn.Sequential,
        ])
    except Exception:
        pass

    if not hasattr(tasks, "_original_parse_model"):
        tasks._original_parse_model = tasks.parse_model
    tasks.parse_model = patched_parse_model


# =====================================================================
# 2. DESCARGA Y CARGA DEL MODELO (best.pt)
# =====================================================================

WEIGHTS_URL = (
    "https://github.com/AnanyaGubba/"
    "GAF-Net-Multi-Disease-Corn-Leaf-Detection/"
    "raw/refs/heads/main/best.pt.zip"
)

# Rutas relativas compatibles con GitHub y servidores web
BASE_DIR = pathlib.Path(__file__).parent.resolve()
MODEL_DIR = BASE_DIR / "gafnet_maize"
INPUT_DIR = BASE_DIR / "gafnet_inputs"
OUTPUT_DIR = BASE_DIR / "gafnet_outputs"

for folder in (MODEL_DIR, INPUT_DIR, OUTPUT_DIR):
    folder.mkdir(parents=True, exist_ok=True)

ZIP_PATH = MODEL_DIR / "best.pt.zip"
BEST_PT = MODEL_DIR / "best.pt"

TRADUCCION_CLASES = {
    "blight": "Tizón foliar (blight)",
    "common_rust": "Roya común (common_rust)",
    "gray_spot": "Mancha gris (gray_spot)",
    "health": "Hoja sana (health)",
}

COLORES_CLASES = {
    "blight": (0, 140, 255),       # Naranja/Rojo en BGR
    "common_rust": (0, 69, 255),   # Rojo fuerte en BGR
    "gray_spot": (180, 105, 255),  # Magenta/Rosa en BGR
    "health": (50, 205, 50),       # Verde en BGR
}


def looks_like_torch_checkpoint_zip(path):
    """Verifica si el archivo ZIP es directamente un checkpoint serializado por PyTorch."""
    if not zipfile.is_zipfile(path):
        return False
    try:
        with zipfile.ZipFile(path, "r") as zf:
            names = zf.namelist()
        has_data_pkl = any(name.endswith("/data.pkl") or name == "data.pkl" for name in names)
        has_tensor_data = any("/data/" in name for name in names)
        return has_data_pkl and has_tensor_data
    except Exception:
        return False


def prepare_weights():
    """Descarga y prepara best.pt si no existe localmente en gafnet_maize/."""
    if BEST_PT.exists() and BEST_PT.stat().st_size > 1_000_000:
        return BEST_PT

    if not ZIP_PATH.exists():
        response = requests.get(WEIGHTS_URL, timeout=120)
        response.raise_for_status()
        ZIP_PATH.write_bytes(response.content)

    if looks_like_torch_checkpoint_zip(ZIP_PATH):
        shutil.copy2(ZIP_PATH, BEST_PT)
    else:
        extract_dir = MODEL_DIR / "weights"
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(ZIP_PATH, "r") as zf:
            zf.extractall(extract_dir)

        pt_files = list(extract_dir.rglob("*.pt"))
        if pt_files:
            shutil.copy2(pt_files[0], BEST_PT)
        else:
            candidates = [p for p in extract_dir.rglob("*") if p.is_file()]
            torch_candidates = [p for p in candidates if looks_like_torch_checkpoint_zip(p)]
            if torch_candidates:
                largest = max(torch_candidates, key=lambda p: p.stat().st_size)
                shutil.copy2(largest, BEST_PT)
            else:
                raise FileNotFoundError("No se pudo identificar automáticamente el checkpoint best.pt.")

    return BEST_PT


@st.cache_resource(show_spinner="Cargando modelo GAF-Net (YOLOv8)...")
def load_gafnet_model():
    register_custom_modules()
    weights_path = prepare_weights()
    model = YOLO(str(weights_path))
    return model


# =====================================================================
# 3. FUNCIONES DE INFERENCIA Y DIBUJO DE CAJAS
# =====================================================================

def procesar_imagen(model, image_pil, filename, conf_thres, iou_thres, img_size):
    """Ejecuta la detección sobre una imagen y devuelve la imagen anotada y la lista de detecciones."""
    register_custom_modules()

    # Convertir PIL (RGB) a OpenCV (BGR)
    img_rgb = np.array(image_pil.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    results = model.predict(
        source=img_bgr,
        conf=conf_thres,
        iou=iou_thres,
        imgsz=img_size,
        verbose=False
    )

    result = results[0]
    annotated_bgr = img_bgr.copy()
    detecciones = []

    boxes = result.boxes
    if boxes is not None and len(boxes) > 0:
        for idx, box in enumerate(boxes, start=1):
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = [int(round(v)) for v in box.xyxy[0].tolist()]

            raw_name = str(model.names.get(cls_id, f"class_{cls_id}")).lower()
            nombre_es = TRADUCCION_CLASES.get(raw_name, raw_name)
            color = COLORES_CLASES.get(raw_name, (255, 165, 0))

            # Dibujar rectángulo
            cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), color, 3)

            # Dibujar etiqueta con fondo
            label = f"{raw_name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            y_text = max(y1 - 8, th + 8)
            cv2.rectangle(
                annotated_bgr,
                (x1, y_text - th - 8),
                (x1 + tw + 8, y_text + 4),
                color,
                -1
            )
            cv2.putText(
                annotated_bgr,
                label,
                (x1 + 4, y_text - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            detecciones.append({
                "Archivo": filename,
                "Detección #": idx,
                "Clase (ID)": cls_id,
                "Código Clase": raw_name,
                "Diagnóstico": nombre_es,
                "Confianza": round(conf, 4),
                "Confianza (%)": f"{conf * 100:.1f}%",
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            })

    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(annotated_rgb), detecciones


# =====================================================================
# 4. INTERFAZ DE USUARIO (STREAMLIT)
# =====================================================================

def main():
    st.set_page_config(
        page_title="GAF-Net | Detección en Maíz",
        page_icon="🌽",
        layout="wide"
    )

    st.title("🌽 GAF-Net: Detección de Enfermedades en Hojas de Maíz")
    st.markdown(
        """
        Aplicación basada en la arquitectura **GAF-Net (YOLOv8 + GSConv + FASFF)** utilizando el checkpoint público `best.pt`.
        * **Clases detectadas:** `blight` (Tizón foliar), `common_rust` (Roya común), `gray_spot` (Mancha gris), `health` (Hoja aparentemente sana).
        """
    )

    # Barra lateral de parámetros
    st.sidebar.header("⚙️ Parámetros de Inferencia")
    conf_thres = st.sidebar.slider(
        "Umbral de confianza (Confidence)",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05,
        help="0.25 = sensible | 0.35 = intermedio | 0.50 = conservador"
    )
    iou_thres = st.sidebar.slider(
        "Umbral IoU para NMS",
        min_value=0.10,
        max_value=0.95,
        value=0.70,
        step=0.05
    )
    img_size = st.sidebar.selectbox(
        "Tamaño de inferencia (imgsz)",
        options=[640, 960, 1280],
        index=0,
        help="Para lesiones pequeñas se recomienda probar 960."
    )

    # Cargar modelo
    try:
        model = load_gafnet_model()
        st.sidebar.success("✅ Modelo GAF-Net cargado correctamente")
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {e}")
        st.stop()

    # Subida de imágenes
    st.subheader("📤 1. Subir fotografías de hojas de maíz")
    uploaded_files = st.file_uploader(
        "Selecciona una o varias imágenes (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if not uploaded_files:
        st.info("👆 Sube una o más fotografías para ejecutar la detección.")
        return

    todas_las_detecciones = []
    imagenes_marcadas = {}

    st.subheader("🔍 2. Resultados del Análisis")

    for uploaded_file in uploaded_files:
        filename = pathlib.Path(uploaded_file.name).name
        image_pil = Image.open(uploaded_file)

        # Guardar copia en gafnet_inputs
        input_save_path = INPUT_DIR / filename
        image_pil.convert("RGB").save(input_save_path)

        with st.spinner(f"Analizando {filename}..."):
            annotated_pil, detecciones = procesar_imagen(
                model=model,
                image_pil=image_pil,
                filename=filename,
                conf_thres=conf_thres,
                iou_thres=iou_thres,
                img_size=img_size
            )

        # Guardar imagen anotada en gafnet_outputs y en memoria para descarga
        output_save_path = OUTPUT_DIR / f"marcada_{filename}"
        annotated_pil.save(output_save_path)

        buf = io.BytesIO()
        annotated_pil.save(buf, format="PNG")
        imagenes_marcadas[f"marcada_{pathlib.Path(filename).stem}.png"] = buf.getvalue()

        todas_las_detecciones.extend(detecciones)

        # Mostrar tarjeta por imagen
        with st.expander(f"📸 Imagen: {filename} ({len(detecciones)} detecciones)", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.image(image_pil, caption="Original", use_container_width=True)
            with col2:
                st.image(annotated_pil, caption="Detección GAF-Net", use_container_width=True)

            if detecciones:
                df_img = pd.DataFrame(detecciones)
                st.dataframe(df_img, use_container_width=True, hide_index=True)
            else:
                st.warning("Ninguna predicción superó el umbral de confianza actual en esta imagen.")

    # Tabla consolidada y descargas
    st.subheader("📊 3. Tabla Consolidada y Descargas")

    if todas_las_detecciones:
        df_total = pd.DataFrame(todas_las_detecciones)
        st.dataframe(df_total, use_container_width=True, hide_index=True)

        csv_bytes = df_total.to_csv(index=False).encode("utf-8-sig")
    else:
        df_total = pd.DataFrame(columns=[
            "Archivo", "Detección #", "Clase (ID)", "Código Clase",
            "Diagnóstico", "Confianza", "Confianza (%)", "x1", "y1", "x2", "y2"
        ])
        st.info("No se registraron detecciones globales con el umbral seleccionado.")
        csv_bytes = df_total.to_csv(index=False).encode("utf-8-sig")

    # Crear archivo ZIP con todas las imágenes marcadas + el CSV
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for img_name, img_bytes in imagenes_marcadas.items():
            zf.writestr(img_name, img_bytes)
        zf.writestr("reporte_detecciones_gafnet.csv", csv_bytes)

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            label="📥 Descargar Tabla Detallada (CSV)",
            data=csv_bytes,
            file_name="reporte_detecciones_gafnet.csv",
            mime="text/csv",
            use_container_width=True
        )
    with col_dl2:
        st.download_button(
            label="📦 Descargar Imágenes Marcadas + CSV (ZIP)",
            data=zip_buffer.getvalue(),
            file_name="resultados_gafnet_maiz.zip",
            mime="application/zip",
            use_container_width=True
        )


if __name__ == "__main__":
    main()