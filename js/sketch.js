const x_vals = []
const y_vals = []

const learning_rate = 0.3
const optimizer = tf.train.sgd(learning_rate)

let lrSlider, pauseButton
let isLooping = true

// slope and y itercept
let m, b, cost = 0

async function setup() {
    createCanvas(windowWidth * 0.7, windowHeight * 0.8)

    // init slope and y intercept
    m = tf.variable(tf.scalar(0))
    b = tf.variable(tf.scalar(0))

    lrSlider = createSlider(0, 1, learning_rate, 0.1)
    pauseButton = createButton("Pause!")

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
}

function draw() {
    background("#333")

    if (x_vals.length) {
        const xs = tf.tensor1d(x_vals)
        const ys = tf.tensor1d(y_vals)

        optimizer.setLearningRate(lrSlider.value())

        cost = optimizer.minimize(() => tf.losses.meanSquaredError(ys, predict(xs)), true).dataSync()

        xs.dispose()
        ys.dispose()
    }else {
        noStroke()
        fill("#999")
        textSize(20)
        text("Click Anywhere!", width * 0.67, height * 0.25)
    }

    drawGraph()
    drawText()
    drawDots()
    drawLine()
}

function predict(x) {
    return x.mul(m).add(b) // y = mx + b
}

// function loss(guess, label) {
//     return guess.sub(label).square().mean() // mean squared error
// }

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
    text(`Learning Rate : ${optimizer.learningRate}`, 2, height - 50)
    text(`Cost : ${cost}`, 2, height - 35)
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
