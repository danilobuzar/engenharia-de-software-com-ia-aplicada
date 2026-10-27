import tf from '@tensorflow/tfjs-node';

// Exemplo de pessoas para treino (cada pessoa com idade, cor e localização)
// const pessoas = [
//     { nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo" },
//     { nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio" },
//     { nome: "Carlos", idade: 40, cor: "verde", localizacao: "Curitiba" }
// ];

// Vetores de entrada com valores já normalizados e one-hot encoded
// Ordem: [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
// const tensorPessoas = [
//     [0.33, 1, 0, 0, 1, 0, 0], // Erick
//     [0, 0, 1, 0, 0, 1, 0],    // Ana
//     [1, 0, 0, 1, 0, 0, 1]     // Carlos
// ]

async function predictPessoa(model, tensorPessoa) {
    const tfInput = tf.tensor2d(tensorPessoa);
    const prediction = model.predict(tfInput);
    const predArray = await prediction.array();

    return predArray[0].map((prob, index) => ({prob, index}));
}

async function trainModel(inputXs, outputYs) {
    // Criamos o modelo sequencial
    const model = tf.sequential();

    model.add(tf.layers.dense({
        inputShape: [7], // 7 features de entrada
        units: 80,       // 80 neurônios na camada oculta
        activation: 'relu' // Função de ativação ReLU
    }));

    model.add(tf.layers.dense({
        units: 3,        // 3 neurônios de saída (para 3 categorias)
        activation: 'softmax' // Função de ativação softmax para classificação
    }));

    model.compile({
        optimizer: 'adam', // Otimizador Adam
        loss: 'categoricalCrossentropy', // Função de perda para classificação multi-classe
        metrics: ['accuracy'] // Métrica de acurácia
    });

    // Treinamos o modelo com os dados de entrada e saída
    await model.fit(inputXs, outputYs, {
        verbose: 0, // Não exibir logs detalhados
        epochs: 100, // Número de épocas de treino
        batchSize: 1, // Tamanho do batch
        shuffle: true, // Embaralhar os dados a cada época
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                console.log(`Epoch ${epoch + 1}: loss = ${logs.loss.toFixed(4)}, accuracy = ${logs.acc.toFixed(4)}`);
            }
        }
    });

    return model;
}

// Usamos apenas os dados numéricos, como a rede neural só entende números.
// tensorPessoasNormalizado corresponde ao dataset de entrada do modelo.
const tensorPessoasNormalizado = [
    [0.33, 1, 0, 0, 1, 0, 0], // Erick
    [0, 0, 1, 0, 0, 1, 0],    // Ana
    [1, 0, 0, 1, 0, 0, 1]     // Carlos
]

// Labels das categorias a serem previstas (one-hot encoded)
// [premium, medium, basic]
const labelsNomes = ["premium", "medium", "basic"]; // Ordem dos labels
const tensorLabels = [
    [1, 0, 0], // premium - Erick
    [0, 1, 0], // medium - Ana
    [0, 0, 1]  // basic - Carlos
];

// Criamos tensores de entrada (xs) e saída (ys) para treinar o modelo
const inputXs = tf.tensor2d(tensorPessoasNormalizado)
const outputYs = tf.tensor2d(tensorLabels)

//inputXs.print();
//outputYs.print();

const model = await trainModel(inputXs, outputYs);

const pessoa = { nome: "João", idade: 28, cor: "verde", localizacao: "Curitiba" };
const tensorPessoa = [[0.2, 1, 0, 0, 1, 0, 0]]; // Normalizado e one-hot encoded
const predicao = await predictPessoa(model, tensorPessoa);
const results = predicao.sort((a, b) => b.prob - a.prob).map(p => `${labelsNomes[p.index]}: ${(p.prob * 100).toFixed(2)}%`).join('\n');
console.log(`Predição para ${pessoa.nome}:\n`, results);