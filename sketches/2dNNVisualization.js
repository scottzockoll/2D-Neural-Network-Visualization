import * as tf from "@tensorflow/tfjs";
import p5 from 'p5'


export const twoDimNeuralNetwork = (canvas_size, model) => {
    const BACKGROUND_COLOR = 250
    const CANVAS_SIZE = canvas_size
    let active_class = 1
    let predictionMode = false
    let xs = []
    let ys = []

    function changeClass() {
        active_class = active_class === 1 ? 0 : 1
    }

    function getCanvasPoints(x, y) {
        return [x + CANVAS_SIZE / 2, CANVAS_SIZE / 2 - y]
    }

    function getCartesianPoints(x, y) {
        return [x - CANVAS_SIZE / 2, CANVAS_SIZE / 2 - y]
    }

    function getArithmeticSequence(res) {
        let difference = (CANVAS_SIZE / res) / (res - 1)
        let result = [0]
        for (let i = 0; i < CANVAS_SIZE; i += difference) {
            if (i !== 0) {
                result.push(result[result.length - 1] + difference)
            }
        }
        return [result, difference]
    }

    const mouseClicked = (s, e) => {
        if (predictionMode) {
            let [x, y] = getCartesianPoints(e.offsetX, e.offsetY)
            model.predict(tf.tensor2d([x, y], [1, 2])).print()
        } else {
            drawPoint(s, e.offsetX, e.offsetY, active_class)
            let [x, y] = getCartesianPoints(e.offsetX, e.offsetY)
            xs.push([x, y])
            ys.push(active_class)
        }
    }

    // x and y are in canvas points
    const drawPoint = (s, x, y, classNumber) => {
        s.stroke(0)
        let color
        if (classNumber === 1) {
            color = 'green'
        } else {
            color = 'red'
        }
        s.fill(color)
        s.circle(x, y, 10)
    }

    const drawLines = (s) => {
        s.stroke(0)
        s.fill(0, 250)
        s.line(CANVAS_SIZE / 2, 0, CANVAS_SIZE / 2, CANVAS_SIZE)
        s.line(0, CANVAS_SIZE / 2, CANVAS_SIZE, CANVAS_SIZE / 2)
    }

    const drawDataset = (s) => {
        s.stroke(0)
        for (let i = 0; i < xs.length; i += 1) {
            let [x1, x2] = getCanvasPoints(...xs[i])
            drawPoint(s, x1, x2, ys[i])
        }
    }

    const clearCanvas = (s) => {
        s.fill(BACKGROUND_COLOR, 250)
        s.noStroke()
        s.rect(0, 0, CANVAS_SIZE, CANVAS_SIZE)
    }

    const clearRegions = (s) => {
        clearCanvas(s)
        drawLines(s)
        drawDataset(s)
    }

    const drawRegions = (s) => {

        let [sequence, difference] = getArithmeticSequence(10)

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

    }

    function train(s, epochs) {
        s.trainButton.attribute('disabled', 'true')
        model.fit(tf.tensor2d(xs, [xs.length, 2]), tf.oneHot(ys, 2), {
            epochs: epochs,
        }).then(() => {
            console.log('Model trained')
            clearRegions(s)
            drawRegions(s)
            s.trainButton.removeAttribute('disabled')
        });

    }


    const sketch = (s) => {

        s.setup = () => {
            // create canvas
            let canvas = s.createCanvas(CANVAS_SIZE, CANVAS_SIZE);
            canvas.parent('p5jsdiv');
            canvas.mouseClicked((e) => mouseClicked(s, e))

            // draw graphics
            s.background(BACKGROUND_COLOR)
            drawLines(s)

            // define buttons
            let changeClassButton = s.createButton('Change class');
            changeClassButton.mousePressed(() => changeClass())
            s.trainButton = s.createButton('Train');
            s.trainButton.mousePressed(() => train(s,300))

        };

        s.keyPressed = (e) => {
            if (e.key === 'p') {
                predictionMode = predictionMode !== true
                console.log(`Prediction mode: ${predictionMode}`)
            }
        }


    };
    new p5(sketch);

}

