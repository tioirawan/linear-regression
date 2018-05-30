const x_vals = []
const y_vals = []

const learning_rate = 0.3
const optimizer = tf.train.sgd(learning_rate)

let lrSlider, pauseButton, resetButton
let isLooping = true

// slope and y itercept
let m, b, loss = 0

async function setup() {
    createCanvas(windowWidth * (windowWidth > 450 ? 0.8 : 0.9), windowHeight * 0.8).parent("canvas-content")

    // init slope and y intercept
    m = tf.variable(tf.scalar(0))
    b = tf.variable(tf.scalar(0))

    lrSlider = select("#lr-slider")
    pauseButton = select("#pause-btn")
    resetButton = select("#reset-btn")

    lrSlider.value(learning_rate)

    pauseButton.mousePressed(() => {
        if (isLooping) {
            noLoop()
            pauseButton.html("Resume!")
        } else {
            loop()
            pauseButton.html("Pause!")
        }

        isLooping = !isLooping
    })

    resetButton.mousePressed(() => {
        x_vals.splice(0, x_vals.length)
        y_vals.splice(0, y_vals.length)
    })
}

function windowResized() {
    resizeCanvas(windowWidth * (windowWidth > 450 ? 0.8 : 0.9), windowHeight * 0.8)
}

function draw() {
    background("#333")

    optimizer.setLearningRate(lrSlider.value())

    if (x_vals.length) {
        tf.tidy(() => {
            const xs = tf.tensor1d(x_vals)
            const ys = tf.tensor1d(y_vals)

            loss = optimizer.minimize(() => tf.losses.meanSquaredError(ys, predict(xs)), true).dataSync()
        })
    } else {
        noStroke()
        fill("#999")
        textSize(20)
        textAlign(CENTER)
        text("Click Anywhere!", width / 2, height * 0.25)
    }

    drawGraph()
    drawText()
    drawDots()
    drawLine()
}

function predict(x) {
    return x.mul(m).add(b) // y = mx + b
}

function mouseClicked() {
    // make sure to just click inside the canvas
    if (mouseX < width && mouseY < height) {
        x_vals.push(normalizeX(mouseX))
        y_vals.push(normalizeY(mouseY))
    }
}

function drawLine() {
    // predict for -1 and 1
    const y = tf.tidy(() => predict(tf.tensor1d([-1, 1])).dataSync())

    const x1 = denormalizeX(-1) // x1 = -1 or 0 width
    const x2 = denormalizeX(1) // x2 = 1 or full width

    const y1 = denormalizeY(y[0]) // y = predict(-1) or predict(x1)
    const y2 = denormalizeY(y[1]) // y = predict(1) or predict(x2)

    stroke("#1dd1a1")
    strokeWeight(1)
    line(x1, y1, x2, y2)
}

function drawText() {
    fill("#999")
    noStroke()
    textSize(15)
    textAlign(LEFT)
    text(`Learning Rate : ${optimizer.learningRate}`, 2, height - 50)
    text(`Loss : ${loss}`, 2, height - 35)
    text(`m : ${m.dataSync()}`, 2, height - 20)
    text(`b : ${b.dataSync()}`, 2, height - 5)
}

function drawDots() {
    const xs = x_vals.map(denormalizeX)
    const ys = y_vals.map(denormalizeY)

    // sorted x data to draw conected points
    const pair = xs.map((x, i) => ({ x, y: ys[i] }))
    pair.sort((a, b) => a.x - b.x)

    for (let i = 0; i < xs.length; i++) {
        // draw loss
        const guess = tf.tidy(() => {
            const x = tf.tensor1d([normalizeX(xs[i])])

            return predict(x).dataSync()
        })
        stroke("#ee5253")
        strokeWeight(1)
        line(xs[i], ys[i], xs[i], denormalizeY(guess))

        // draw line connection
        if (pair[i + 1]) {
            stroke("#404040")
            strokeWeight(0.5)
            line(pair[i].x, pair[i].y, pair[i + 1].x, pair[i + 1].y)
        }

        // draw dot points
        stroke("#2e86de")
        strokeWeight(7)
        point(xs[i], ys[i])
    }
}

function drawGraph() {
    stroke("#999")
    strokeWeight(1)
    line(width / 2, 0, width / 2, height)
    line(0, height / 2, width, height / 2)
}
