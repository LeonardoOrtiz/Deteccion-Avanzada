#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

# Configurar estilo
plt.style.use('dark_background')

# ============================================
# Gráfico 1: LOF
# ============================================
print("Generando gráfico LOF")
X = np.array([[1,1], [1.2,1.1], [0.8,1], [8,8]])
modelo_lof = LocalOutlierFactor(n_neighbors=2, contamination=0.25)
predicciones_lof = modelo_lof.fit_predict(X)

fig, ax = plt.subplots(figsize=(8, 6))
colores = ['#2ecc71' if p == 1 else '#e74c3c' for p in predicciones_lof]
ax.scatter(X[:, 0], X[:, 1], c=colores, s=300, alpha=0.8, edgecolors='white', linewidth=2)
ax.set_xlabel('Característica 1', fontsize=12, fontweight='bold')
ax.set_ylabel('Característica 2', fontsize=12, fontweight='bold')
ax.set_title('Detección de Anomalías con LOF', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ecc71', markersize=12, label='Normal'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', markersize=12, label='Anomalía')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=11)
plt.tight_layout()
plt.savefig('images/lof_plot.png', dpi=150, bbox_inches='tight')
print("Guardado: images/lof_plot.png")
plt.close()

# ============================================
# Gráfico 2: One-Class SVM
# ============================================
print("Generando gráfico One-Class SVM")
X_normal = np.array([[1,1], [1.1,1], [0.9,1.2], [1,0.8], [1.05,1.05]])
modelo_svm = OneClassSVM(kernel="rbf", gamma=0.5, nu=0.05)
modelo_svm.fit(X_normal)

X_test = np.array([[1,1], [10,10], [0.95,1.1], [5,5]])
predicciones_svm = modelo_svm.predict(X_test)

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(X_normal[:, 0], X_normal[:, 1], c='#3498db', s=200, alpha=0.8, 
           label='Entrenamiento (Normal)', edgecolors='white', linewidth=2)
colores_test = ['#2ecc71' if p == 1 else '#e74c3c' for p in predicciones_svm]
ax.scatter(X_test[:, 0], X_test[:, 1], c=colores_test, s=300, alpha=0.8, marker='X',
           label='Predicciones', edgecolors='white', linewidth=2)
ax.set_xlabel('Característica 1', fontsize=12, fontweight='bold')
ax.set_ylabel('Característica 2', fontsize=12, fontweight='bold')
ax.set_title('One-Class SVM: Frontera de Decisión', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right', fontsize=11)
plt.tight_layout()
plt.savefig('images/svm_plot.png', dpi=150, bbox_inches='tight')
print("Guardado: images/svm_plot.png")
plt.close()