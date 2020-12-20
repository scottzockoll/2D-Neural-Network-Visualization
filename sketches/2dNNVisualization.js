import * as tf from "@tensorflow/tfjs";
import p5 from 'p5'


export const twoDimNeuralNetwork = (canvas_size, model) => {
    const CANVAS_SIZE = canvas_size
    let active_class = 1
    let predictionMode = false
    let xs = []
    let ys = []

    function changeClass() {
        active_class = active_class === 1 ? 0 : 1
    }

    function getCartesianPoints(x, y) {
        return [x - CANVAS_SIZE / 2, CANVAS_SIZE / 2 - y]
    }

    function getSequence(res) {
        let difference = (CANVAS_SIZE / res) / (res - 1)
        let result = []
        for (let i = 0; i < CANVAS_SIZE; i += difference) {
            if (i === 0) {
                result.push(0)
            } else {
                result.push(result[result.length - 1] + difference)
            }
        }
        return [result, difference]
    }

    const mouseClicked = (s, e) => {
        if (predictionMode) {
            let [x, y] = getCartesianPoints(e.offsetX, e.offsetY)
            console.log(x, y)
            model.predict(tf.tensor2d([x, y], [1, 2])).print()
        } else {
            let color
            if (active_class === 1) {
                color = 'green'
            } else {
                color = 'red'
            }
            s.fill(color)
            let [x, y] = getCartesianPoints(e.offsetX, e.offsetY)
            xs.push([x, y])
            ys.push(active_class)
            console.log(x, y)
            s.circle(e.offsetX, e.offsetY, 10)
        }
    }

    const drawRegions = (s) => {

        let [sequence, difference] = getSequence(10)

        let cartesianMesh = []
        let canvasMesh = []
        sequence.forEach((y_component) => {
            sequence.forEach((x_component) => {
                let [x, y] = getCartesianPoints(x_component, y_component)
                cartesianMesh.push([x, y])
                canvasMesh.push([x_component, y_component])
            })
        })

        let predictions
        model.predict(tf.tensor2d(cartesianMesh, [cartesianMesh.length, 2])).array().then((arrayFromPromise) => {

            predictions = arrayFromPromise

            for (let i = 0; i < predictions.length; i += 1) {
                let prediction = predictions[i].indexOf(Math.max(...predictions[i]))
                let [x, y] = canvasMesh[i]
                if (prediction === 1) {
                    s.fill(0, 250, 0, 100)
                } else {
                    s.fill(250, 0, 0, 100)
                }
                s.noStroke()
                s.rect(x, y, difference, difference)
            }
        })

        model.predict(tf.tensor2d([[0, 0]], [1, 2])).print()

    }

    function train(epochs) {
        model.fit(tf.tensor2d(xs, [xs.length, 2]), tf.oneHot(ys, 2), {
            epochs: epochs,
        }).then(() => {
            console.log('Model trained')
        });
    }

    const sketch = (s) => {


        s.setup = () => {
            let canvas = s.createCanvas(CANVAS_SIZE, CANVAS_SIZE);
            canvas.parent('p5jsdiv');
            canvas.mouseClicked((e) => mouseClicked(s, e))
            s.background(150)
            s.line(CANVAS_SIZE / 2, 0, CANVAS_SIZE / 2, CANVAS_SIZE)
            s.line(0, CANVAS_SIZE / 2, CANVAS_SIZE, CANVAS_SIZE / 2)
            let changeClassButton = s.createButton('Change class');
            changeClassButton.mousePressed(() => changeClass())
            let trainButton = s.createButton('Train');
            trainButton.mousePressed(() => train(500))
            let drawRegionsButton = s.createButton('Draw regions')
            drawRegionsButton.mousePressed(() => drawRegions(s))

        };

        s.keyPressed = (e) => {
            console.log(s)
            if (e.key === 'p') {
                predictionMode = predictionMode !== true
                console.log(`Prediction mode: ${predictionMode}`)
            }
        }


    };
    new p5(sketch);

}

