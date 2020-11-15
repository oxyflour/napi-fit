const { interp1 } = require('./lib/utils'),
    fs = require('fs')

function makeSrc(proj, dt, tn) {
    const tree = "1D Results\\Port signals\\i1",
        [sx, sy] = [0, 1].map(type => proj.get1DResult(tree, 0, type === 0 ? 0 : 1)),
        src = new Float32Array(tn)
    for (let i = 0; i < tn; i ++) {
        src[i] = interp1(sx, sy, i * dt * 1e9);
    }
    return src
}

const { fit: { Solver }, occ: { Mesher }, cst: { Project } } = require('.'),
    proj = new Project('E:\\Projects\\cst-demo\\dipole-test.cst', 2019),
    grid = proj.getHexGrid(),
    { dt, tn } = proj.getMeta(),
    src = makeSrc(proj, dt, tn),
    mats = { eps: proj.getMatrix(100), mue: proj.getMatrix(101) },
    solver = new Solver(grid, mats, dt)

console.time('run')
console.log(`dt ${dt * 1e9} ns, ${tn} time steps`)
const csv = []
for (let i = 0; i < tn; i ++) {
    const s = solver.step(src[i])
    csv.push(`${i},${src[i]},${s}`)
}
fs.writeFileSync('build/plot.csv', csv.join('\n'))
console.timeEnd('run')

proj.destroy()
solver.destroy()
