# 21/10/2025

###################################################################################################
#####                        FS COMPLEXITY BASED, DISTRIBUTED VERSION                         #####
###################################################################################################

# En este script vamos a programar una manera de seleccionar variables un poco similar al RF
# que es de forma distribuida. Básicamente lo que vamos a hacer es un muestreo de variables
# Según el SOTA lo mejor es hacer un muestreo con reemplazamiento porque si no hay variables
# que nunca se estudian conjuntamente y se pierde esa interrelación

# La idea es la siguiente:
# Boostrap de variables: tomamos n réplicas de m random variables with replacement
# Evaluar cada réplica de forma multivariante (aquí podemos jugar con correlación, quizás merece la pena quitar primero las correladas)
# En base a la info de las n réplicas, construir un gráfico de importancia (tipo RF) que me diga
# cuánto disminuye la complejidad cada variable. Para ello, como necesito un punto de partida,
# cojo las m variables de cada réplica, calculo la complejidad y ya tengo el punto de partida.
# Ahora comienzo a quitar variables tipo backward selection y voy apuntando  lo que se va disminuyendo de complejidad
# Esto lo puedo hacer de forma aleatoria (opción 1) o guiándonos por la complejidad univariante de cada variable (opción 2)
# porque se supone que cuanto menor sea la complejidad con una variable, mejor es dicha variable.
# Con esto obtengo un gráfico de importancia de las variables
# La idea es ir metiendo variables tipo forward en base a esas variables y para cuando ya no disminuyan la complejidad




