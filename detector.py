import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import numpy as np
from scipy.spatial import distance as dist
import cv2


class DetectorDesnutricionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Detector de Desnutrición Facial")
        self.root.geometry("1000x650")
        self.root.configure(bg="#f0f0f0")

        # Variables
        self.imagen_path = None
        self.imagen_original = None
        self.imagen_resultado = None

        # Crear interfaz
        self.crear_widgets()

        # Pre-cargar clasificadores
        try:
            self.detector_rostro = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.detector_ojos = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            if self.detector_rostro.empty() or self.detector_ojos.empty():
                messagebox.showerror("Error",
                                     "No se pudieron cargar los clasificadores. Verifique la instalación de OpenCV.")
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar clasificadores: {str(e)}")

    def crear_widgets(self):
        # Frame principal con dos columnas
        frame_principal = tk.Frame(self.root, bg="#f0f0f0")
        frame_principal.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Frame izquierdo (controles)
        self.frame_izquierdo = tk.Frame(frame_principal, bg="#f0f0f0", width=300)
        self.frame_izquierdo.pack(side=tk.LEFT, fill=tk.BOTH, padx=10, pady=10)

        # Título
        tk.Label(self.frame_izquierdo, text="Detector de Desnutrición Facial",
                 font=("Arial", 14, "bold"), bg="#f0f0f0").pack(pady=10)

        # Botón para cargar imagen - Usando Entry + Button
        frame_carga = tk.Frame(self.frame_izquierdo, bg="#f0f0f0")
        frame_carga.pack(fill=tk.X, pady=10)

        self.ruta_entry = tk.Entry(frame_carga, width=20)
        self.ruta_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        tk.Button(frame_carga, text="Examinar...", command=self.buscar_imagen).pack(side=tk.RIGHT)

        # Botón para analizar
        self.btn_analizar = tk.Button(self.frame_izquierdo, text="Analizar Imagen",
                                      command=self.iniciar_analisis, width=15,
                                      state=tk.DISABLED, bg="#4CAF50", fg="white")
        self.btn_analizar.pack(pady=10)

        # Área de resultados
        tk.Label(self.frame_izquierdo, text="Resultados del Análisis:",
                 font=("Arial", 12, "bold"), bg="#f0f0f0").pack(anchor=tk.W, pady=(20, 5))

        # Frame para resultados con scroll
        frame_resultados = tk.Frame(self.frame_izquierdo, bd=1, relief=tk.SUNKEN)
        frame_resultados.pack(fill=tk.BOTH, expand=True, pady=5)

        scrollbar = tk.Scrollbar(frame_resultados)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.texto_resultados = tk.Text(frame_resultados, height=15, width=30,
                                        yscrollcommand=scrollbar.set, wrap=tk.WORD)
        self.texto_resultados.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.texto_resultados.yview)

        # Frame derecho (visualización)
        self.frame_derecho = tk.Frame(frame_principal, bg="#e0e0e0")
        self.frame_derecho.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Label para mostrar la imagen
        self.lbl_imagen = tk.Label(self.frame_derecho, bg="#d0d0d0", text="Cargue una imagen para comenzar")
        self.lbl_imagen.pack(fill=tk.BOTH, expand=True)

        # Barra de estado
        self.barra_estado = tk.Label(self.root, text="Listo para comenzar",
                                     bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.barra_estado.pack(side=tk.BOTTOM, fill=tk.X)

    def buscar_imagen(self):
        """Abre diálogo para seleccionar un archivo de imagen"""
        try:
            # Mostrar cuadro de diálogo para seleccionar archivo
            filename = filedialog.askopenfilename(
                title="Seleccionar imagen facial",
                filetypes=[
                    ("Imágenes", "*.jpg *.jpeg *.png *.bmp *.gif"),
                    ("Todos los archivos", "*.*")
                ]
            )

            # Si se seleccionó un archivo
            if filename:
                self.ruta_entry.delete(0, tk.END)
                self.ruta_entry.insert(0, filename)
                self.cargar_imagen(filename)
        except Exception as e:
            messagebox.showerror("Error", f"Error al abrir el diálogo de archivos: {str(e)}")

    def cargar_imagen(self, ruta):
        """Carga y muestra la imagen seleccionada"""
        try:
            # Cargar imagen usando OpenCV
            self.imagen_path = ruta
            self.imagen_original = cv2.imread(ruta)

            if self.imagen_original is None:
                messagebox.showerror("Error", f"No se pudo leer la imagen: {ruta}")
                return

            # Convertir de BGR a RGB para mostrar
            imagen_rgb = cv2.cvtColor(self.imagen_original, cv2.COLOR_BGR2RGB)

            # Redimensionar para mostrar
            self.mostrar_imagen(imagen_rgb, self.lbl_imagen)

            # Habilitar análisis
            self.btn_analizar.config(state=tk.NORMAL)
            self.barra_estado.config(text=f"Imagen cargada: {os.path.basename(ruta)}")

            # Limpiar resultados anteriores
            self.texto_resultados.delete(1.0, tk.END)

        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar la imagen: {str(e)}")

    def mostrar_imagen(self, imagen, label, max_size=500):
        """Redimensiona y muestra una imagen en un Label"""
        h, w = imagen.shape[:2]

        # Calcular nueva dimensión manteniendo proporción
        if h > w:
            nueva_h = min(h, max_size)
            factor = nueva_h / h
            nueva_w = int(w * factor)
        else:
            nueva_w = min(w, max_size)
            factor = nueva_w / w
            nueva_h = int(h * factor)

        # Redimensionar imagen
        img_redim = cv2.resize(imagen, (nueva_w, nueva_h), interpolation=cv2.INTER_AREA)

        # Convertir a formato para Tkinter
        img_pil = Image.fromarray(img_redim)
        img_tk = ImageTk.PhotoImage(image=img_pil)

        # Actualizar label
        label.config(image=img_tk, text="")
        label.image = img_tk  # Mantener referencia

    def iniciar_analisis(self):
        """Analiza la imagen cargada"""
        if self.imagen_original is None:
            messagebox.showwarning("Advertencia", "Primero debe cargar una imagen")
            return

        try:
            self.barra_estado.config(text="Analizando rostro...")
            self.root.update()

            # Analizar la imagen
            resultado = self.analizar_nutricion_facial(self.imagen_original)

            if resultado:
                # Mostrar imagen con anotaciones
                imagen_rgb = cv2.cvtColor(resultado["imagen_resultado"], cv2.COLOR_BGR2RGB)
                self.mostrar_imagen(imagen_rgb, self.lbl_imagen)

                # Mostrar resultados en texto
                self.mostrar_resultados(resultado)

                if resultado["posible_desnutricion"]:
                    self.barra_estado.config(text="⚠️ Posibles signos de desnutrición detectados")
                else:
                    self.barra_estado.config(text="✓ No se detectaron signos de desnutrición")
            else:
                self.barra_estado.config(text="No se pudo completar el análisis")

        except Exception as e:
            messagebox.showerror("Error", f"Error durante el análisis: {str(e)}")
            self.barra_estado.config(text="Error en el análisis")

    def mostrar_resultados(self, resultado):
        """Muestra los resultados del análisis en el área de texto"""
        self.texto_resultados.delete(1.0, tk.END)

        # Mostrar métricas
        self.texto_resultados.insert(tk.END, "MÉTRICAS FACIALES:\n", "titulo")
        self.texto_resultados.insert(tk.END, f"• Ancho rostro: {resultado['ancho_rostro']:.1f} px\n")
        self.texto_resultados.insert(tk.END, f"• Distancia entre ojos: {resultado['distancia_interpupilar']:.1f} px\n")
        self.texto_resultados.insert(tk.END, f"• Ratio ojos/rostro: {resultado['ratio_ojos_rostro']:.3f}\n")
        self.texto_resultados.insert(tk.END, f"• Ancho mejillas: {resultado['ancho_mejillas']:.1f} px\n")
        self.texto_resultados.insert(tk.END, f"• Ratio mejillas/ojos: {resultado['ratio_mejillas_ojos']:.3f}\n\n")

        # Mostrar evaluación
        self.texto_resultados.insert(tk.END, "EVALUACIÓN:\n", "titulo")

        if resultado["posible_desnutricion"]:
            self.texto_resultados.insert(tk.END, "⚠️ POSIBLES SIGNOS DE DESNUTRICIÓN:\n", "alerta")
            for signo in resultado["signos_desnutricion"]:
                self.texto_resultados.insert(tk.END, f"  - {signo}\n")

            self.texto_resultados.insert(tk.END,
                                         "\nNOTA: Este análisis es preliminar y no reemplaza una evaluación médica profesional.",
                                         "nota")
        else:
            self.texto_resultados.insert(tk.END,
                                         "✓ No se detectaron signos evidentes de desnutrición según los parámetros analizados.",
                                         "normal")

        # Configurar estilos de texto
        self.texto_resultados.tag_configure("titulo", font=("Arial", 10, "bold"))
        self.texto_resultados.tag_configure("alerta", foreground="red")
        self.texto_resultados.tag_configure("nota", foreground="blue", font=("Arial", 9, "italic"))
        self.texto_resultados.tag_configure("normal", foreground="green")

    def analizar_nutricion_facial(self, imagen):
        """
        Analiza una imagen facial para detectar posibles signos de desnutrición
        """
        # Convertir a escala de grises
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

        # Detectar rostros
        rostros = self.detector_rostro.detectMultiScale(gris, scaleFactor=1.1, minNeighbors=5)

        if len(rostros) == 0:
            messagebox.showinfo("Información", "No se detectó ningún rostro en la imagen")
            return None

        # Trabajar con el primer rostro
        x, y, w, h = rostros[0]

        # Región de interés (ROI) - Rostro
        roi_gris = gris[y:y + h, x:x + w]

        # Detectar ojos en el rostro
        ojos = self.detector_ojos.detectMultiScale(roi_gris, scaleFactor=1.1, minNeighbors=5)

        if len(ojos) < 2:
            messagebox.showinfo("Información", "No se pudieron detectar ambos ojos claramente")
            return None

        # Ordenar ojos de izquierda a derecha
        ojos_ordenados = sorted(ojos, key=lambda ojo: ojo[0])

    #
        if len(ojos_ordenados) > 2:
            # Primero por tamaño (área)
            ojos_ordenados = sorted(ojos_ordenados, key=lambda ojo: ojo[2] * ojo[3], reverse=True)[:2]
            # Luego por posición horizontal
            ojos_ordenados = sorted(ojos_ordenados, key=lambda ojo: ojo[0])

        ojo_izq = ojos_ordenados[0]
        ojo_der = ojos_ordenados[1]

        # Calcular centros de los ojos
        centro_ojo_izq = (x + int(ojo_izq[0] + ojo_izq[2] / 2), y + int(ojo_izq[1] + ojo_izq[3] / 2))
        centro_ojo_der = (x + int(ojo_der[0] + ojo_der[2] / 2), y + int(ojo_der[1] + ojo_der[3] / 2))

        # Calcular métricas
        # 1. Distancia interpupilar (entre ojos)
        distancia_interpupilar = dist.euclidean(
            (centro_ojo_izq[0], centro_ojo_izq[1]),
            (centro_ojo_der[0], centro_ojo_der[1])
        )

        # 2. Ancho del rostro
        ancho_rostro = w

        # 3. Ratio entre distancia interpupilar y ancho del rostro
        ratio_ojos_rostro = distancia_interpupilar / ancho_rostro

        # 4. Calcular la posición estimada de las mejillas
        altura_mejillas = int(y + h * 2 / 3)
        pos_mejilla_izq = (x, altura_mejillas)
        pos_mejilla_der = (x + w, altura_mejillas)

        # 5. Anchura a nivel de mejillas
        ancho_mejillas = dist.euclidean(pos_mejilla_izq, pos_mejilla_der)

        # 6. Ratio entre ancho de mejillas y distancia interpupilar
        ratio_mejillas_ojos = ancho_mejillas / distancia_interpupilar

        # Crear imagen con resultados visuales
        imagen_resultado = imagen.copy()

        # Dibujar rectángulo del rostro
        cv2.rectangle(imagen_resultado, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Dibujar rectángulos de los ojos
        for (ex, ey, ew, eh) in ojos_ordenados:
            cv2.rectangle(imagen_resultado, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)

        # Dibujar centros de los ojos
        cv2.circle(imagen_resultado, centro_ojo_izq, 5, (0, 0, 255), -1)
        cv2.circle(imagen_resultado, centro_ojo_der, 5, (0, 0, 255), -1)

        # Dibujar línea interpupilar
        cv2.line(imagen_resultado, centro_ojo_izq, centro_ojo_der, (0, 255, 255), 2)

        # Dibujar línea de mejillas
        cv2.line(imagen_resultado, pos_mejilla_izq, pos_mejilla_der, (255, 255, 0), 2)

        # Añadir texto informativo
        cv2.putText(imagen_resultado, f"Dist. ojos: {distancia_interpupilar:.1f}px",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(imagen_resultado, f"Ancho rostro: {ancho_rostro:.1f}px",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Definir umbrales (aproximados)
        umbral_ratio_ojos_rostro = 0.35  # Si es mayor, rostro estrecho
        umbral_ratio_mejillas_ojos = 3.0  # Si es menor, mejillas hundidas

        # Evaluación de posibles signos
        signos_desnutricion = []

        if ratio_ojos_rostro > umbral_ratio_ojos_rostro:
            signos_desnutricion.append("Rostro estrecho en proporción a los ojos")

        if ratio_mejillas_ojos < umbral_ratio_mejillas_ojos:
            signos_desnutricion.append("Posible hundimiento de mejillas")

        # Resultado final
        resultado = {
            "ancho_rostro": ancho_rostro,
            "distancia_interpupilar": distancia_interpupilar,
            "ratio_ojos_rostro": ratio_ojos_rostro,
            "ancho_mejillas": ancho_mejillas,
            "ratio_mejillas_ojos": ratio_mejillas_ojos,
            "signos_desnutricion": signos_desnutricion,
            "posible_desnutricion": len(signos_desnutricion) > 0,
            "imagen_resultado": imagen_resultado
        }

        return resultado


def main():
    # Iniciar aplicación
    root = tk.Tk()
    app = DetectorDesnutricionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()