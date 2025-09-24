##### 24/09/2025
#### En este script vamos a calcular (hacemos las funciones para) la complejidad de distintos conjuntos de datos
#### en diferentes versiones: todas las variables, solo las informativas, solo las redundantes o ruidosa,
### un mix de ellas, las que seleccionan algunos métodos de filtro de FS del estado del artee
### La idea es ver cómo cambia el comportamiento de las medidas de complejidad en esas circunstancias
### para evaluar, un poco a priori, si las podemos utilizar para crear un method de FS basado en
#### medidas de complejidad. en el notebook TrackingCentroides_Hostility hemos visto que las variables que son
### literalmente la copia de otras, son fáciles de pillar porque muestran el mismo comportamiento.
### Sin embargo, las que son redundantes por ser combinación lineal de otras, ya no se ve claramente cómo pillarlas
### Leyendo el SOTA veo que ese es uno de los principales problemas de los métodos de filtro.