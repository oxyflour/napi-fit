#include "fit.h"
#include "utils.h"

#include <regex>
#include <iostream>

template <typename T> void copyFromArray(Napi::TypedArrayOf<T> arr, int &len, T *&buf) {
    len = arr.ElementLength();
    buf = new T[len];
    for (int i = 0; i < len; i ++) {
        buf[i] = arr[i];
    }
}

Napi::FunctionReference FitSolver::constructor;
void FitSolver::Init(Napi::Env env, Napi::Object exports) {
    Napi::HandleScope scope(env);
    Napi::Function func = DefineClass(env, "Solver", {
        InstanceMethod("destroy", &FitSolver::Destroy),
        InstanceMethod("step", &FitSolver::Step),
    });
    constructor = Napi::Persistent(func);
    constructor.SuppressDestruct();
    exports.Set("Solver", func);
}

FitSolver::FitSolver(const Napi::CallbackInfo& info) : Napi::ObjectWrap<FitSolver>(info) {
    auto mesh = info[0].As<Napi::Array>();
    copyFromArray(mesh.Get("xs").As<Napi::Float64Array>(), nx, xs);
    copyFromArray(mesh.Get("ys").As<Napi::Float64Array>(), ny, ys);
    copyFromArray(mesh.Get("zs").As<Napi::Float64Array>(), nz, zs);
    auto nvar = nx * ny * nz * 3;

    auto mat = info[1].As<Napi::Object>();
    auto eps = mat.Get("eps").As<Napi::Float32Array>(),
        mue = mat.Get("mue").As<Napi::Float32Array>();
    auto dt = info[2].ToNumber().DoubleValue();

    // TODO: move this out
    float kap = 0, rho = 0;
    le = new float[nvar];
    re = new float[nvar];
    lh = new float[nvar];
    rh = new float[nvar];
    for (int i = 0; i < nvar; i ++) {
        auto ep = eps[i], mu = mue[i];
        le[i] = (1 - kap * ep * dt / 2) / (1 + kap * ep * dt / 2);
        re[i] = dt * ep / (1 + kap * ep * dt / 2);
        lh[i] = (1 - rho * mu * dt / 2) / (1 + rho * mu * dt / 2);
        rh[i] = dt * mu / (1 + rho * mu * dt / 2);
    }

    // TODO: move this out
    auto root = dirname(dirname(__FILE__));
    auto chunkSrc = readFile(root + "/tpl/chunk.cu");
    chunkSrc = std::regex_replace(chunkSrc, std::regex("_\\$i(\\W)"), "_0$1");
    chunkSrc = std::regex_replace(chunkSrc, std::regex("\\$nx(\\W)"), std::to_string(nx) + "$1");
    chunkSrc = std::regex_replace(chunkSrc, std::regex("\\$ny(\\W)"), std::to_string(ny) + "$1");
    chunkSrc = std::regex_replace(chunkSrc, std::regex("\\$nz(\\W)"), std::to_string(nz) + "$1");
    chunkSrc = std::regex_replace(chunkSrc, std::regex("\\$sg(\\W)"), std::to_string(-1) + "$1");
    chunkSrc = std::regex_replace(chunkSrc, std::regex("\\$sd(\\W)"), std::to_string(0) + "$1");

    auto postSrc = readFile(root + "/tpl/post.cu");
    writeFile(root + "/build/tpl.cu", chunkSrc + postSrc);

    auto inc = root + "/include",
        src = root + "/build/tpl.cu",
        out = root + "/build/tpl.dll",
        log = root + "/build/tpl.log",
        cmd = "nvcc -I" + inc + " " + src + " --shared -o " + out +" > " + log + " 2>&1";
    auto ret = system(cmd.c_str());
    if (ret == 0) {
        dll = OPENLIB(utf8_to_wstring(out).c_str());
        if (dll != NULL) {
            fInit = (init_ptr) LIBFUNC(dll, "init_0");
            fStep = (step_ptr) LIBFUNC(dll, "step_0");
            fQuit = (quit_ptr) LIBFUNC(dll, "quit_0");
            auto ret = fInit ? fInit(le, re, lh, rh) : -1;
            if (ret == 0) {
                // pass
            } else {
                Napi::Error::New(info.Env(), "call init failed")
                    .ThrowAsJavaScriptException();
            }
        } else {
            Napi::Error::New(info.Env(), "load dll failed: build/tpl.dll")
                .ThrowAsJavaScriptException();
        }
    } else {
        std::cerr << readFile(log);
        Napi::Error::New(info.Env(), "compile failed: code " + std::to_string(ret) + ", cmd: " + cmd)
            .ThrowAsJavaScriptException();
    }
}

Napi::Value FitSolver::Destroy(const Napi::CallbackInfo& info) {
    delete le;
    delete re;
    delete lh;
    delete rh;
    delete xs;
    delete ys;
    delete zs;
    if (fQuit) {
        fQuit();
    }
    return info.Env().Null();
}

Napi::Value FitSolver::Step(const Napi::CallbackInfo& info) {
    auto s = info[0].ToNumber().FloatValue();
    if (fStep != NULL) {
        s = fStep(s);
        return Napi::Number::New(info.Env(), s);
    } else {
        Napi::Error::New(info.Env(), "load dll failed: build/tpl.dll")
            .ThrowAsJavaScriptException();
        return info.Env().Null();
    }
}

Napi::Object InitFit(Napi::Env env, Napi::Object exports) {
    auto fit = Napi::Object::New(env);
    FitSolver::Init(env, fit);
    exports.Set(Napi::String::New(env, "fit"), fit);
    return exports;
}
