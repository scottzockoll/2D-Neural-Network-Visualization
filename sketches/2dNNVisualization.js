import * as tf from "@tensorflow/tfjs";
import p5 from 'p5'


export const twoDimNeuralNetwork = (canvas_size, model) => {
    const BACKGROUND_COLOR = 250
    const CANVAS_SIZE = canvas_size
    let active_class = 1
    let predictionMode = false
    let xs = []
    let ys = []
    let dataGraphic
    let hasDrawnRegions = false
    let cartesianMesh
    let canvasMesh
    let sequence
    let difference
    let region
    let regionStrokes

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
            drawPoint(dataGraphic, e.offsetX, e.offsetY, active_class)
            let [x, y] = getCartesianPoints(e.offsetX, e.offsetY)
            xs.push([x, y])
            ys.push(active_class)
            s.image(dataGraphic, 0, 0)
        }
    }

    // x and y are in canvas points
    // draws point onto graphic
    const drawPoint = (graphic, x, y, classNumber) => {
        graphic.stroke(0)
        let color
        if (classNumber === 1) {
            color = 'green'
        } else {
            color = 'red'
        }
        graphic.fill(color)
        graphic.circle(x, y, 10)
    }

    // draw lines onto this graphic
    const drawLines = (graphic) => {
        graphic.stroke(0)
        graphic.fill(0, 250)
        graphic.line(CANVAS_SIZE / 2, 0, CANVAS_SIZE / 2, CANVAS_SIZE)
        graphic.line(0, CANVAS_SIZE / 2, CANVAS_SIZE, CANVAS_SIZE / 2)
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

        if (!hasDrawnRegions) {
            console.log('computing region info')
            hasDrawnRegions = true;
            [sequence, difference] = getArithmeticSequence(10)
            cartesianMesh = []
            canvasMesh = []
            region = []
            // regionStrokes = []
            sequence.forEach((y_component) => {
                sequence.forEach((x_component) => {
                    let [x, y] = getCartesianPoints(x_component, y_component)
                    cartesianMesh.push([x, y])
                    canvasMesh.push([x_component, y_component])
                    region.push(2)
                    // regionStrokes.push(0)
                })
            })

        }

        let predictions
        model.predict(tf.tensor2d(cartesianMesh, [cartesianMesh.length, 2])).array().then((arrayFromPromise) => {

            predictions = arrayFromPromise

            for (let i = 0; i < predictions.length; i += 1) {
                let prediction = predictions[i].indexOf(Math.max(...predictions[i]))
                if (prediction !== region[i]) {
                    let [x, y] = canvasMesh[i]
                    s.noStroke()
                    s.fill(BACKGROUND_COLOR, 250)
                    s.rect(x, y, difference, difference)
                    if (prediction === 1) {
                        s.fill(0, 250, 0, 100)
                        region[i] = 1
                    } else {
                        s.fill(250, 0, 0, 100)
                        region[i] = 0
                    }
                    // s.stroke(0)
                    // regionStrokes[i] = 1
                    s.rect(x, y, difference, difference)
                    s.image(dataGraphic, 0, 0)
                }
                // else if (regionStrokes[i] === 1) {
                //     regionStrokes[i] = 0
                //     let [x, y] = canvasMesh[i]
                //     s.noStroke()
                //     s.fill(BACKGROUND_COLOR, 250)
                //     s.rect(x, y, difference, difference)
                //     let prediction = predictions[i].indexOf(Math.max(...predictions[i]))
                //     if (prediction === 1) {
                //         s.fill(0, 250, 0, 100)
                //     } else {
                //         s.fill(250, 0, 0, 100)
                //     }
                //     s.noStroke()
                //     s.rect(x, y, difference, difference)
                //     s.image(dataGraphic, 0, 0)
                // }
            }
        })

    }

    function train(s, epochs) {
        s.trainButton.attribute('disabled', 'true')
        model.fit(tf.tensor2d(xs, [xs.length, 2]), tf.oneHot(ys, 2), {
            epochs: epochs,
            callbacks: {
                onEpochEnd: (epoch) => {
                    // console.log(epoch)
                    drawRegions(s)
                    // if (epoch % 3 === 0) {
                    //     console.log(epoch)
                        // clearRegions(s)
                        // drawRegions(s)
                    // }

                }
            }
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
            dataGraphic = s.createGraphics(CANVAS_SIZE, CANVAS_SIZE)
            dataGraphic.background(0, 0)
            drawLines(dataGraphic)
            s.image(dataGraphic, 0, 0)

            // define buttons
            let changeClassButton = s.createButton('Change class');
            changeClassButton.mousePressed(() => changeClass())
            s.trainButton = s.createButton('Train');
            s.trainButton.mousePressed(() => train(s, 300))

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

