class Solver {
    constructor(grid: {
        xs: Float64Array
        ys: Float64Array
        zs: Float64Array
    }, mats: {
        eps: Float32Array
        mue: Float32Array
    }, dt: number)
    destroy(): void
    step(s: number): number
}

export const fit = { Solver }

class Mesher {
    constructor(xs: Float64Array, ys: Float64Array, zs: Float64Array)
    getMats(shape: Shape): { eps: Float32Array, mue: Float32Array }
}

export const occ = { Mesher }

class Project {
    constructor(public file: string, public version: string | number)
    destroy(): void
    getHexGrid(): { xs: Float64Array, ys: Float64Array, zs: Float64Array }
    getMatrix(type: 100 | 101): Float32Array
    get1DResult(tree: string, num?: number, type?: 0 | 1): Float32Array
    getMeta(): {
        dt: number
        tn: number
    }
}

export const cst = { Project }
