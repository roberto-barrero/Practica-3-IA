from collections import defaultdict
import math

datos_entrenamiento = [
    {
        "Pronóstico": "Soleado",
        "Temperatura": 36,
        "Humedad": "Alta",
        "Viento": "Leve",
        "Asado": "No"
    },
    {
        "Pronóstico": "Soleado",
        "Temperatura": 28,
        "Humedad": "Alta",
        "Viento": "Fuerte",
        "Asado": "No"
    },
    {
        "Pronóstico": "Nublado",
        "Temperatura": 30,
        "Humedad": "Alta",
        "Viento": "Leve",
        "Asado": "Sí"
    },
    {
        "Pronóstico": "Lluvioso",
        "Temperatura": 20,
        "Humedad": "Alta",
        "Viento": "Leve",
        "Asado": "Sí"
    },
    {
        "Pronóstico": "Lluvioso",
        "Temperatura": 2,
        "Humedad": "Normal",
        "Viento": "Leve",
        "Asado": "Sí"
    },
    {
        "Pronóstico": "Lluvioso",
        "Temperatura": 5,
        "Humedad": "Normal",
        "Viento": "Fuerte",
        "Asado": "No"
    },
    {
        "Pronóstico": "Nublado",
        "Temperatura": 11,
        "Humedad": "Normal",
        "Viento": "Fuerte",
        "Asado": "Sí"
    },
    {
        "Pronóstico": "Soleado",
        "Temperatura": 22,
        "Humedad": "Alta",
        "Viento": "Leve",
        "Asado": "No"
    },
    {
        "Pronóstico": "Soleado",
        "Temperatura": 9,
        "Humedad": "Normal",
        "Viento": "Leve",
        "Asado": "Sí"
    },
    {
        "Pronóstico": "Lluvioso",
        "Temperatura": 17,
        "Humedad": "Normal",
        "Viento": "Leve",
        "Asado": "Sí"
    },
    {
        "Pronóstico": "Soleado",
        "Temperatura": 19,
        "Humedad": "Normal",
        "Viento": "Fuerte",
        "Asado": "Sí"
    },
    {
        "Pronóstico": "Nublado",
        "Temperatura": 22,
        "Humedad": "Alta",
        "Viento": "Fuerte",
        "Asado": "Sí"
    },
    {
        "Pronóstico": "Nublado",
        "Temperatura": 27,
        "Humedad": "Normal",
        "Viento": "Leve",
        "Asado": "Sí"
    },
    {
        "Pronóstico": "Lluvioso",
        "Temperatura": 21,
        "Humedad": "Alta",
        "Viento": "Fuerte",
        "Asado": "No"
    }
]

datos_inferencia = [
    {
        "Pronóstico": "Soleado",
        "Temperatura": 19,
        "Humedad": "Normal",
        "Viento": "Leve"
    },
    {
        "Pronóstico": "Lluvioso",
        "Temperatura": 34,
        "Humedad": "Alta",
        "Viento": "Leve"
    },
    {
        "Pronóstico": "Nublado",
        "Temperatura": 14,
        "Humedad": "Normal",
        "Viento": "Fuerte"
    }
]


class NaiveBayesClassifier:
    # Clase para clasificar datos usando el algoritmo Naive Bayes
    def __init__(self, className='class'):
        self.class_counts = defaultdict(int)
        self.feature_counts_per_class = defaultdict(lambda: defaultdict(int))
        self.feature_count_global = defaultdict(int)
        self.feature_probabilities = defaultdict(lambda: defaultdict(float))
        self.class_probabilities = defaultdict(lambda: defaultdict(float))
        self.classes = set()
        self.features = set()
        self.className = className
        self.training_data = None

    # Calcula la probabilidad de un valor suponiendo una distribucion normal
    def calculate_gaussian_probability(self, feature, value, class_):
        entries = [
            entry for entry in self.training_data if entry["Asado"] == class_]
        mean = 0
        for entry in entries:
            mean += entry[feature]
        mean /= float(self.class_counts[class_])

        variance = sum([pow(x[feature]-mean, 2) for x in entries]) / \
            float(self.class_counts[class_]-1)
        stdev = math.sqrt(variance)

        # print(f"Mean: {mean}, Stdev: {stdev}")D

        return 1/(math.sqrt(2*math.pi)*math.pow(stdev, 2))*math.exp(-math.pow(value-mean, 2)/(2*math.pow(stdev, 2)))

    # Obtiene la probabilidad de un valor dado un feature y una clase
    def get_feature_probability(self, feature, value, class_):
        if type(value) == int or type(value) == float:
            return self.calculate_gaussian_probability(feature, value, class_)
        return self.feature_probabilities[class_].get((feature, value), 0.01)

    # Entrena el clasificador con los datos
    def train(self, data):
        # Calcular numero de veces que aparece cada clase
        self.training_data = data
        for entry in data:
            # Agregar clase a clases
            self.class_counts[entry[self.className]] += 1
            self.classes.add(entry[self.className])

            # Agregar parametros a feature_counts y contar apariciones
            for feature, value in entry.items():
                if feature != self.className:
                    self.features.add(feature)
                    if type(value) == int or type(value) == float:
                        self.feature_counts_per_class[entry[self.className]
                                                      ][feature, value] = 0
                        self.feature_count_global[feature, value] = 0
                    else:
                        self.feature_counts_per_class[entry[self.className]
                                                      ][feature, value] += 1
                        self.feature_count_global[feature, value] += 1

        # Calcular probabilidades de cada clase y de cada feature
        for class_ in self.classes:
            # Calcular probabilidad de clase
            self.class_probabilities[class_] = self.class_counts[class_] / \
                len(data)
            # Calcular probabilidad condicional de cada parametro
            for feature, value in self.feature_counts_per_class[class_].keys():
                if type(value) == int or type(value) == float:
                    self.feature_probabilities[class_][feature,
                                                       value] = self.get_feature_probability(feature, value, class_)
                else:
                    self.feature_probabilities[class_][feature,
                                                       value] = self.feature_counts_per_class[class_][feature, value] / self.class_counts[class_]
    # Predice la clase de una entrada

    def predict(self, entry):
        predicted_probabilities = {}
        for class_ in self.classes:
            predicted_probabilities[class_] = self.class_probabilities[class_]
            for feature, value in entry.items():
                if feature != self.className:
                    predicted_probabilities[class_] *= self.get_feature_probability(
                        feature, value, class_)
            print(
                f"Probabilidad de {self.className} = {class_}: {predicted_probabilities[class_]}")

        return max(predicted_probabilities, key=predicted_probabilities.get)

    # Imprime las probabilidades de cada clase y de cada feature
    def __print__(self):
        print("Class probabilities:\n")
        total_class_probabilities = 0
        for class_, probability in self.class_probabilities.items():
            print(f"{class_}: {probability}\n")
            total_class_probabilities += probability
        print("Total class probabilities:", total_class_probabilities)
        print("Feature probabilities:")
        for class_, feature_probabilities in self.feature_probabilities.items():
            print(f"{class_}:\n")
            for feature, probability in feature_probabilities.items():
                print(f"  {feature}: {probability}\n")


if __name__ == '__main__':

    # Entrenar con los datos
    classifier = NaiveBayesClassifier("Asado")
    classifier.train(datos_entrenamiento)

    # Inferir con los datos
    resultados = []
    for entry in datos_inferencia:
        print("\n")
        # Imprimir Con feature = value, por cada feature
        resultado = "Con "
        for feature, value in entry.items():
            resultado += f"{feature} = {value}, "
        resultado += f"el asado {classifier.predict(entry)} se hace."
        print(resultado)
