const tfvis = require('@tensorflow/tfjs-vis')
const tf = require('@tensorflow/tfjs')

const getData = async () => {
    const result = await fetch('https://storage.googleapis.com/tfjs-tutorials/carsData.json')
    const parse = await result.json()
    const finalResult = parse.map(car => ({
        mpg: car.Miles_per_Gallon,
        horsepower: car.Horsepower,
    }))
        .filter(car => (car.mpg != null && car.horsepower != null));

    return finalResult
}

const run = async () => {
    const data = await getData()
    const values = data.map(d => ({
        x: d.horsepower,
        y: d.mpg,
    }));
    tfvis.render.scatterplot(
        { name: 'Horsepower v MPG' },
        { values },
        {
            xLabel: 'Horsepower',
            yLabel: 'MPG',
            height: 300
        }
    );
    const model = createModel();
    tfvis.show.modelSummary({ name: 'Model Summary' }, model);

    const tensorData = convertToTensor(data)
    const {inputs,labels} = tensorData
    
    await trainModel(model,inputs,labels);
    
    testModel(model,data,tensorData)
}

const createModel = () => {
    const model = tf.sequential();

    model.add(tf.layers.dense({ inputShape: [1], units: 1, useBias: true }))

    model.add(tf.layers.dense({ units: 1, useBias: true }))
    return model
}

const convertToTensor = (data) => {

    return tf.tidy(() => {
        tf.util.shuffle(data)

        const inputs = data.map(d => d.horsepower)
        const labels = data.map(d => d.mpg)
        const inputTensors = tf.tensor2d(inputs, [inputs.length, 1])
        const labelTensors = tf.tensor2d(labels, [labels.length, 1])

        const inputMax = inputTensors.max()
        const inputMin = inputTensors.min()
        const labelMax = labelTensors.max()
        const labelMin = labelTensors.min()

        const normalizedInputs = inputTensors.sub(inputMin).div(inputMax.sub(inputMin)) // (x-min(x)) / (max(x) - min(x)) absolute formula  
        const normalizedLabels = labelTensors.sub(labelMin).div(labelMax.sub(labelMin))
        
        return {
            inputs: normalizedInputs,
            labels: normalizedLabels,
            inputMax,
            inputMin,
            labelMax,
            labelMin
        }
    })
}

const trainModel = async (model, inputs, labels) => {
    model.compile({
        optimizer: tf.train.adam(),
        loss: tf.losses.meanSquaredError,
        metrics: ['mse']
    })

    const batchSize = 28
    const epochs = 200
    return await model.fit(inputs, labels, {
        batchSize,
        epochs,
        shuffle: true,
        callbacks: tfvis.show.fitCallbacks(
            { name: 'Training Performance' },
            ['loss', 'mse'],
            { height: 200, callbacks: ['onEpochEnd'] }
        )
    })
}

const testModel = (model,inputData,normalizationData) => {
    const {inputMax, inputMin, labelMin, labelMax} = normalizationData;  

    const [xs, preds] = tf.tidy(() => {
      
      const xs = tf.linspace(0, 1, 100);      
      const preds = model.predict(xs.reshape([100, 1]));      
      
      const unNormXs = xs
        .mul(inputMax.sub(inputMin))
        .add(inputMin);
      
      const unNormPreds = preds
        .mul(labelMax.sub(labelMin))
        .add(labelMin);
      

      return [unNormXs.dataSync(), unNormPreds.dataSync()];
    });
    
    const predictedPoints = Array.from(xs).map((val, i) => {
      return {x: val, y: preds[i]}
    });
    
    const originalPoints = inputData.map(d => ({
      x: d.horsepower, y: d.mpg,
    }));
    
    tfvis.render.scatterplot(
      {name: 'Model Predictions vs Original Data'}, 
      {values: [originalPoints, predictedPoints], series: ['original', 'predicted']}, 
      {
        xLabel: 'Horsepower',
        yLabel: 'MPG',
        height: 300
      }
    );
}


document.addEventListener('DOMContentLoaded', run);