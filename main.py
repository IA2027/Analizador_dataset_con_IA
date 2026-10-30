
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from dotenv import load_dotenv

import pandas as pd
import numpy as np

from google import genai

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor
)

from sklearn.linear_model import LogisticRegression, LinearRegression

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score
)

from sklearn.cluster import KMeans


# ==========================================
# CONFIGURACION
# ==========================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

MODEL = "gemini-3.5-flash-lite"

if API_KEY:
    client = genai.Client(api_key=API_KEY)
else:
    client = None


# ==========================================
# APLICACION
# ==========================================

class ZeroDatasetAI:

    def __init__(self, root):

        self.root = root

        self.root.title("DATASET ANALYZER AI")

        self.root.geometry("1100x750")

        self.root.minsize(900, 650)

        self.root.configure(bg="#171B24")

        self.df = None
        self.archivo = None

        self.crear_interfaz()

    # ======================================
    # INTERFAZ
    # ======================================

    def crear_interfaz(self):

        estilo = ttk.Style()

        estilo.theme_use("clam")

        estilo.configure(
            "TCombobox",
            fieldbackground="#252D3B",
            background="#252D3B",
            foreground="white"
        )

        titulo = tk.Label(
            self.root,
            text="IA DATASET ANALYZER",
            font=("Segoe UI", 23, "bold"),
            bg="#171B24",
            fg="#FF4545"
        )

        titulo.pack(pady=(20, 5))

        subtitulo = tk.Label(
            self.root,
            text="INTELIGENCIA ARTIFICIAL + MACHINE LEARNING",
            font=("Segoe UI", 11),
            bg="#171B24",
            fg="#55B9FF"
        )

        subtitulo.pack(pady=(0, 20))

        # PANEL SUPERIOR

        panel = tk.Frame(
            self.root,
            bg="#222A38",
            padx=15,
            pady=15
        )

        panel.pack(fill="x", padx=25)

        self.btn_cargar = tk.Button(
            panel,
            text="CARGAR DATASET",
            command=self.cargar_dataset,
            bg="#1769AA",
            fg="white",
            font=("Segoe UI", 11, "bold"),
            padx=15,
            pady=10,
            relief="flat",
            cursor="hand2"
        )

        self.btn_cargar.pack(side="left", padx=5)

        self.lbl_archivo = tk.Label(
            panel,
            text="Ningun dataset seleccionado",
            bg="#222A38",
            fg="white",
            font=("Segoe UI", 10)
        )

        self.lbl_archivo.pack(side="left", padx=15)

        # SELECCION TARGET

        target_panel = tk.Frame(
            self.root,
            bg="#171B24"
        )

        target_panel.pack(fill="x", padx=30, pady=15)

        tk.Label(
            target_panel,
            text="COLUMNA OBJETIVO (TARGET):",
            bg="#171B24",
            fg="white",
            font=("Segoe UI", 10, "bold")
        ).pack(side="left", padx=5)

        self.combo_target = ttk.Combobox(
            target_panel,
            state="readonly",
            width=30
        )

        self.combo_target.pack(side="left", padx=10)

        self.combo_target.set("Sin target / No supervisado")

        # BOTON ANALISIS

        self.btn_analizar = tk.Button(
            self.root,
            text="ANALIZAR DATASET CON IA",
            command=self.analizar_dataset,
            bg="#087E65",
            fg="white",
            font=("Segoe UI", 13, "bold"),
            padx=20,
            pady=12,
            relief="flat",
            cursor="hand2"
        )

        self.btn_analizar.pack(pady=10)

        # ESTADO

        self.estado = tk.Label(
            self.root,
            text="Esperando dataset...",
            bg="#171B24",
            fg="#55B9FF",
            font=("Segoe UI", 10)
        )

        self.estado.pack(pady=5)

        # RESULTADOS

        tk.Label(
            self.root,
            text="INFORME DE ANALISIS",
            bg="#171B24",
            fg="#FF4545",
            font=("Segoe UI", 13, "bold")
        ).pack(pady=(10, 5))

        contenedor = tk.Frame(
            self.root,
            bg="#222A38"
        )

        contenedor.pack(
            fill="both",
            expand=True,
            padx=25,
            pady=(0, 20)
        )

        self.resultado = tk.Text(
            contenedor,
            bg="#10151E",
            fg="#FFFFFF",
            insertbackground="white",
            font=("Consolas", 10),
            wrap="word",
            relief="flat",
            padx=15,
            pady=15
        )

        scroll = ttk.Scrollbar(
            contenedor,
            command=self.resultado.yview
        )

        self.resultado.configure(
            yscrollcommand=scroll.set
        )

        self.resultado.pack(
            side="left",
            fill="both",
            expand=True
        )

        scroll.pack(
            side="right",
            fill="y"
        )

    # ======================================
    # CARGAR DATASET
    # ======================================

    def cargar_dataset(self):

        archivo = filedialog.askopenfilename(

            title="Seleccionar dataset",

            filetypes=[
                ("Archivos CSV y Excel",
                 "*.csv *.xlsx *.xls"),
                ("CSV", "*.csv"),
                ("Excel", "*.xlsx *.xls")
            ]

        )

        if not archivo:
            return

        try:

            if archivo.lower().endswith(".csv"):

                try:
                    df = pd.read_csv(
                        archivo,
                        sep=None,
                        engine="python",
                        encoding="utf-8"
                    )

                except UnicodeDecodeError:

                    df = pd.read_csv(
                        archivo,
                        sep=None,
                        engine="python",
                        encoding="latin-1"
                    )

            else:

                df = pd.read_excel(archivo)

            if df.empty:
                raise ValueError("El dataset esta vacio.")

            if len(df.columns) < 2:
                raise ValueError(
                    "El dataset necesita al menos dos columnas."
                )

            self.df = df
            self.archivo = archivo

            self.lbl_archivo.config(
                text=os.path.basename(archivo)
            )

            self.combo_target["values"] = [
                "Sin target / No supervisado"
            ] + list(df.columns)

            self.combo_target.current(0)

            self.resultado.delete("1.0", tk.END)

            self.resultado.insert(
                tk.END,
                "DATASET CARGADO CORRECTAMENTE\n\n"
            )

            self.resultado.insert(
                tk.END,
                f"Archivo: {os.path.basename(archivo)}\n"
            )

            self.resultado.insert(
                tk.END,
                f"Filas: {df.shape[0]}\n"
            )

            self.resultado.insert(
                tk.END,
                f"Columnas: {df.shape[1]}\n\n"
            )

            self.resultado.insert(
                tk.END,
                "COLUMNAS DETECTADAS:\n\n"
            )

            for columna in df.columns:

                self.resultado.insert(
                    tk.END,
                    f"- {columna}: {df[columna].dtype}\n"
                )

            self.estado.config(
                text="Dataset listo para analizar."
            )

        except Exception as e:

            messagebox.showerror(
                "Error al cargar dataset",
                str(e)
            )

    # ======================================
    # INFORMACION ESTADISTICA
    # ======================================

    def obtener_estadisticas(self):

        df = self.df

        info = {

            "filas": int(df.shape[0]),

            "columnas": int(df.shape[1]),

            "duplicados": int(
                df.duplicated().sum()
            ),

            "valores_faltantes": int(
                df.isnull().sum().sum()
            ),

            "columnas_numericas": [],

            "columnas_categoricas": [],

            "detalle_columnas": {}

        }

        for columna in df.columns:

            serie = df[columna]

            if (
                pd.api.types.is_numeric_dtype(serie)
                and not pd.api.types.is_bool_dtype(serie)
            ):

                info["columnas_numericas"].append(
                    columna
                )

            else:

                info["columnas_categoricas"].append(
                    columna
                )

            info["detalle_columnas"][columna] = {

                "tipo": str(serie.dtype),

                "valores_unicos": int(
                    serie.nunique()
                ),

                "faltantes": int(
                    serie.isnull().sum()
                )

            }

            if pd.api.types.is_numeric_dtype(serie):

                info["detalle_columnas"][columna][
                    "promedio"
                ] = (
                    round(float(serie.mean()), 3)
                    if serie.notna().any()
                    else None
                )

        return info

    # ======================================
    # DETERMINAR TIPO DE PROBLEMA
    # ======================================

    def determinar_problema(self, target):

        serie = self.df[target].dropna()

        if serie.nunique() < 2:

            raise ValueError(
                "La columna target necesita al menos dos clases o valores."
            )

        if (
            pd.api.types.is_numeric_dtype(serie)
            and not pd.api.types.is_bool_dtype(serie)
            and serie.nunique() > 20
            and serie.nunique() / len(serie) > 0.05
        ):

            return "REGRESION"

        return "CLASIFICACION"

    # ======================================
    # PREPROCESAMIENTO
    # ======================================

    def crear_preprocesador(self, X):

        numericas = X.select_dtypes(
            include=np.number
        ).columns.tolist()

        categoricas = [
            col for col in X.columns
            if col not in numericas
        ]

        transformadores = []

        if numericas:

            pipeline_num = Pipeline([

                (
                    "imputer",
                    SimpleImputer(strategy="median")
                ),

                (
                    "scaler",
                    StandardScaler()
                )

            ])

            transformadores.append(
                ("numericas", pipeline_num, numericas)
            )

        if categoricas:

            pipeline_cat = Pipeline([

                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent")
                ),

                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False
                    )
                )

            ])

            transformadores.append(
                ("categoricas", pipeline_cat, categoricas)
            )

        if not transformadores:

            raise ValueError(
                "No existen variables utilizables."
            )

        return ColumnTransformer(
            transformadores,
            remainder="drop"
        )

    # ======================================
    # COMPARAR ALGORITMOS
    # ======================================

    def comparar_algoritmos(self, target, problema):

        df = self.df.copy()

        df = df.replace([np.inf, -np.inf], np.nan)

        df = df.dropna(subset=[target])

        X = df.drop(columns=[target])

        y = df[target]

        # Evitar columnas completamente vacias

        X = X.dropna(axis=1, how="all")

        # Evitar identificadores unicos

        ids = [
            col for col in X.columns
            if X[col].nunique(dropna=True) == len(X)
        ]

        X = X.drop(columns=ids)

        if X.empty:
            raise ValueError(
                "No quedan variables predictoras utilizables."
            )

        if len(X) < 20:
            raise ValueError(
                "Se necesitan al menos 20 registros para comparar modelos."
            )

        # Limitar datos para evitar consumo excesivo

        if len(X) > 10000:

            X = X.sample(
                10000,
                random_state=42
            )

            y = y.loc[X.index]

        preprocesador = self.crear_preprocesador(X)

        if problema == "CLASIFICACION":

            if y.value_counts().min() < 2:

                raise ValueError(
                    "Cada clase necesita al menos dos registros."
                )

            estratificar = (
                y if y.value_counts().min() >= 2
                else None
            )

            X_train, X_test, y_train, y_test = (
                train_test_split(
                    X,
                    y,
                    test_size=0.25,
                    random_state=42,
                    stratify=estratificar
                )
            )

            modelos = {

                "Logistic Regression":
                    LogisticRegression(max_iter=2000),

                "Random Forest Classifier":
                    RandomForestClassifier(
                        n_estimators=150,
                        random_state=42,
                        n_jobs=-1
                    )

            }

        else:

            if y.nunique() < 2:
                raise ValueError(
                    "La variable objetivo no tiene variabilidad."
                )

            X_train, X_test, y_train, y_test = (
                train_test_split(
                    X,
                    y,
                    test_size=0.25,
                    random_state=42
                )
            )

            modelos = {

                "Linear Regression":
                    LinearRegression(),

                "Random Forest Regressor":
                    RandomForestRegressor(
                        n_estimators=150,
                        random_state=42,
                        n_jobs=-1
                    )

            }

        resultados = {}

        for nombre, modelo in modelos.items():

            try:

                pipeline = Pipeline([

                    ("preprocesamiento", preprocesador),

                    ("modelo", modelo)

                ])

                pipeline.fit(
                    X_train,
                    y_train
                )

                predicciones = pipeline.predict(
                    X_test
                )

                if problema == "CLASIFICACION":

                    resultados[nombre] = {

                        "Accuracy": round(
                            accuracy_score(
                                y_test,
                                predicciones
                            ), 4
                        ),

                        "F1": round(
                            f1_score(
                                y_test,
                                predicciones,
                                average="weighted",
                                zero_division=0
                            ), 4
                        )

                    }

                else:

                    resultados[nombre] = {

                        "MAE": round(
                            mean_absolute_error(
                                y_test,
                                predicciones
                            ), 4
                        ),

                        "RMSE": round(
                            float(
                                np.sqrt(
                                    mean_squared_error(
                                        y_test,
                                        predicciones
                                    )
                                )
                            ), 4
                        ),

                        "R2": round(
                            r2_score(
                                y_test,
                                predicciones
                            ), 4
                        )

                    }

            except Exception as e:

                resultados[nombre] = {
                    "error": str(e)
                }

        validos = {
            k: v for k, v in resultados.items()
            if "error" not in v
        }

        if not validos:

            raise ValueError(
                "No fue posible entrenar los modelos."
            )

        if problema == "CLASIFICACION":

            mejor = max(
                validos,
                key=lambda k: validos[k]["F1"]
            )

        else:

            mejor = max(
                validos,
                key=lambda k: validos[k]["R2"]
            )

        return resultados, mejor

    # ======================================
    # APRENDIZAJE NO SUPERVISADO
    # ======================================

    def analizar_no_supervisado(self):

        X = self.df.select_dtypes(
            include=np.number
        ).copy()

        X = X.replace(
            [np.inf, -np.inf],
            np.nan
        )

        X = X.dropna(axis=1, how="all")

        if X.shape[1] < 2 or len(X) < 10:

            return {
                "recomendacion": "PCA o análisis exploratorio",
                "motivo": (
                    "Se necesitan al menos dos variables "
                    "numéricas y diez registros."
                )
            }

        X = X.sample(
            min(5000, len(X)),
            random_state=42
        )

        X = X.fillna(
            X.median()
        )

        X = StandardScaler().fit_transform(X)

        if not np.isfinite(X).all():

            return {
                "recomendacion": "PCA",
                "motivo": "Hay datos no válidos para clustering."
            }

        resultados = {}

        for k in range(2, min(6, len(X) - 1)):

            try:

                modelo = KMeans(
                    n_clusters=k,
                    random_state=42,
                    n_init=10
                )

                etiquetas = modelo.fit_predict(X)

                if len(np.unique(etiquetas)) > 1:

                    score = silhouette_score(
                        X,
                        etiquetas
                    )

                    resultados[k] = round(
                        score, 4
                    )

            except Exception:
                continue

        if resultados:

            mejor_k = max(
                resultados,
                key=resultados.get
            )

            return {

                "recomendacion": "K-Means",

                "clusters": mejor_k,

                "silhouette": resultados[mejor_k],

                "resultados": resultados

            }

        return {

            "recomendacion": "PCA",

            "motivo": (
                "No se encontró una partición "
                "de clusters evaluable."
            )

        }

    # ======================================
    # CONSULTAR GEMINI
    # ======================================

    def consultar_gemini(
        self,
        estadisticas,
        target,
        problema,
        resultados
    ):

        if client is None:

            return (
                "No se encontró GEMINI_API_KEY.\n"
                "El análisis local está disponible, "
                "pero no se pudo consultar Gemini."
            )

        prompt = f"""
Actúa como científico de datos senior
especializado en Machine Learning.

Analiza el siguiente dataset:

ESTADISTICAS:
{estadisticas}

COLUMNA OBJETIVO:
{target}

TIPO DE PROBLEMA:
{problema}

RESULTADOS DE LOS ALGORITMOS:
{resultados}

Tu trabajo es generar un informe técnico
pero fácil de entender.

Incluye:

1. Diagnóstico del dataset.
2. Tipo de problema de Machine Learning.
3. Mejor algoritmo recomendado.
4. Explicación de por qué lo recomiendas.
5. Interpretación de las métricas.
6. Algoritmos alternativos.
7. Ventajas y desventajas.
8. Posibles problemas de sobreajuste.
9. Preprocesamiento recomendado.
10. Próximos pasos para mejorar el modelo.

IMPORTANTE:

- No inventes métricas.
- No afirmes que un modelo es el mejor
  universalmente.
- Diferencia resultados medidos de hipótesis.
- Considera tamaño y calidad del dataset.
- Advierte si los datos son insuficientes.
- Si hay riesgo de fuga de información,
  indícalo.
- Utiliza español.
"""

        try:

            response = client.models.generate_content(
                model=MODEL,
                contents=prompt
            )

            return response.text or (
                "Gemini no devolvió texto."
            )

        except Exception as e:

            return (
                "ERROR DE GEMINI:\n\n"
                + str(e)
            )

    # ======================================
    # ANALISIS PRINCIPAL
    # ======================================

    def analizar_dataset(self):

        if self.df is None:

            messagebox.showwarning(
                "Atencion",
                "Primero debes cargar un dataset."
            )

            return

        self.btn_analizar.config(
            state="disabled"
        )

        self.estado.config(
            text="Analizando dataset..."
        )

        self.resultado.delete(
            "1.0",
            tk.END
        )

        self.root.update_idletasks()

        try:

            estadisticas = self.obtener_estadisticas()

            seleccion = self.combo_target.get()

            self.resultado.insert(
                tk.END,
                "========== DIAGNOSTICO ==========\n\n"
            )

            self.resultado.insert(
                tk.END,
                f"Filas: {estadisticas['filas']}\n"
            )

            self.resultado.insert(
                tk.END,
                f"Columnas: {estadisticas['columnas']}\n"
            )

            self.resultado.insert(
                tk.END,
                f"Duplicados: {estadisticas['duplicados']}\n"
            )

            self.resultado.insert(
                tk.END,
                f"Valores faltantes: "
                f"{estadisticas['valores_faltantes']}\n\n"
            )

            self.resultado.insert(
                tk.END,
                "========== ALGORITMOS ==========\n\n"
            )

            if seleccion == "Sin target / No supervisado":

                problema = "NO SUPERVISADO"

                target = "No definido"

                resultados = self.analizar_no_supervisado()

                self.resultado.insert(
                    tk.END,
                    f"Recomendacion local: "
                    f"{resultados['recomendacion']}\n\n"
                )

            else:

                target = seleccion

                problema = self.determinar_problema(
                    target
                )

                resultados, mejor = self.comparar_algoritmos(
                    target,
                    problema
                )

                self.resultado.insert(
                    tk.END,
                    f"Tipo: {problema}\n\n"
                )

                for nombre, metricas in resultados.items():

                    self.resultado.insert(
                        tk.END,
                        f"{nombre}\n"
                    )

                    self.resultado.insert(
                        tk.END,
                        f"{metricas}\n\n"
                    )

                self.resultado.insert(
                    tk.END,
                    f"MEJOR RESULTADO MEDIDO: {mejor}\n\n"
                )

            self.resultado.insert(
                tk.END,
                "========== ANALISIS GEMINI ==========\n\n"
            )

            self.root.update_idletasks()

            informe = self.consultar_gemini(
                estadisticas,
                target,
                problema,
                resultados
            )

            self.resultado.insert(
                tk.END,
                informe
            )

            self.estado.config(
                text="Analisis finalizado."
            )

        except Exception as e:

            messagebox.showerror(
                "Error de analisis",
                str(e)
            )

            self.estado.config(
                text="Error durante el analisis."
            )

        finally:

            self.btn_analizar.config(
                state="normal"
            )


# ==========================================
# EJECUTAR
# ==========================================

if __name__ == "__main__":

    root = tk.Tk()

    app = ZeroDatasetAI(root)

    root.mainloop()
