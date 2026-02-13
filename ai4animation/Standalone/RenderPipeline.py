# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import re
from dataclasses import dataclass
from sys import platform
from typing import Any

import cffi
import numpy as np
from pyray import Color, Rectangle, RenderTexture, Texture, Vector2, Vector3
from raylib import (
    BeginMode3D,
    BeginShaderMode,
    BeginTextureMode,
    BLACK,
    CAMERA_ORTHOGRAPHIC,
    CAMERA_PERSPECTIVE,
    ClearBackground,
    DEG2RAD,
    DrawModelEx,
    DrawTextureRec,
    EndMode3D,
    EndShaderMode,
    EndTextureMode,
    GetScreenHeight,
    GetScreenWidth,
    GetShaderLocation,
    LoadRenderTexture,
    LoadShader as LoadShaderFromFile,
    LoadShaderFromMemory,
    MatrixInvert,
    MatrixLookAt,
    MatrixMultiply,
    MatrixToFloatV,
    PIXELFORMAT_UNCOMPRESSED_R16G16B16A16,
    PIXELFORMAT_UNCOMPRESSED_R8G8B8A8,
    RAYWHITE,
    RL_ATTACHMENT_COLOR_CHANNEL0,
    RL_ATTACHMENT_COLOR_CHANNEL1,
    RL_ATTACHMENT_DEPTH,
    RL_ATTACHMENT_TEXTURE2D,
    RL_MODELVIEW,
    RL_PROJECTION,
    rlActiveDrawBuffers,
    rlActiveTextureSlot,
    rlDisableColorBlend,
    rlDisableDepthTest,
    rlDisableFramebuffer,
    rlDrawRenderBatchActive,
    rlEnableColorBlend,
    rlEnableDepthTest,
    rlEnableFramebuffer,
    rlEnableShader,
    rlEnableTexture,
    rlFramebufferAttach,
    rlFramebufferComplete,
    rlFrustum,
    rlGetCullDistanceFar,
    rlGetCullDistanceNear,
    rlGetMatrixModelview,
    rlGetMatrixProjection,
    rlLoadFramebuffer,
    rlLoadIdentity,
    rlLoadTexture,
    rlLoadTextureDepth,
    rlMatrixMode,
    rlMultMatrixf,
    rlOrtho,
    rlPopMatrix,
    rlPushMatrix,
    rlSetClipPlanes,
    rlSetFramebufferHeight,
    rlSetFramebufferWidth,
    rlSetUniform,
    rlUnloadFramebuffer,
    rlViewport,
    SetShaderValue,
    SetShaderValueMatrix,
    SetShaderValueTexture,
    SHADER_UNIFORM_FLOAT,
    SHADER_UNIFORM_INT,
    SHADER_UNIFORM_VEC2,
    SHADER_UNIFORM_VEC3,
    UnloadRenderTexture,
    UnloadShader,
    Vector3Normalize,
    Vector3Scale,
    Vector3Zero,
    WHITE,
)

ffi = cffi.FFI()
from ai4animation import Utility as utils
from ai4animation.Components.Component import Component
from ai4animation.Standalone.SkinnedMesh import SkinnedMesh


# A wrapper around LoadShader to optionally overwrite the glsl version to support OSX
def LoadShader(vertex_shader_name, fragment_shader_name):
    shader_dir = os.path.join(utils.GetDirectory(__file__), "resources", "shaders")
    vertex_shader_path = os.path.join(shader_dir, vertex_shader_name)
    fragment_shader_path = os.path.join(shader_dir, fragment_shader_name)

    # Mac has deprecated OpenGL support
    # Attempt to patch the opengl version but there are no guarantees
    if platform == "darwin":
        if not os.path.isfile(vertex_shader_path):
            raise FileNotFoundError(f"Vertex shader not found {vertex_shader_path}")
        if not os.path.isfile(fragment_shader_path):
            raise FileNotFoundError(f"Fragment shader not found {fragment_shader_path}")
        with open(vertex_shader_path, "r") as f:
            vs = f.read()
        with open(fragment_shader_path, "r") as f:
            fs = f.read()

        OSX_GLSL_VERSION = "410"
        WarningMsg = f"WARNING: Attempting to patch shader {vertex_shader_name}/{fragment_shader_name} to GLSL {OSX_GLSL_VERSION} for OSX"
        print(WarningMsg)
        vs = re.sub("#version.*", f"#version {OSX_GLSL_VERSION}", vs)
        fs = re.sub("#version.*", f"#version {OSX_GLSL_VERSION}", fs)
        return LoadShaderFromMemory(utils.ToBytes(vs), utils.ToBytes(fs))
    else:
        return LoadShaderFromFile(
            utils.ToBytes(vertex_shader_path), utils.ToBytes(fragment_shader_path)
        )


@dataclass
class RegisteredModel:
    """Represents a model registered for rendering."""

    name: str
    model: Any
    skinned_mesh: SkinnedMesh
    position: Vector3
    rotationAxis: Vector3
    rotationAngle: float
    scale: Vector3
    color: Color

    def Draw(self, shader):
        if isinstance(self.model, list):
            for m in self.model:
                for i in range(m.materialCount):
                    m.materials[i].shader = shader
                DrawModelEx(
                    m,
                    self.position,
                    self.rotationAxis,
                    self.rotationAngle,
                    self.scale,
                    self.color,
                )
        else:
            for i in range(self.model.materialCount):
                self.model.materials[i].shader = shader
            DrawModelEx(
                self.model,
                self.position,
                self.rotationAxis,
                self.rotationAngle,
                self.scale,
                self.color,
            )


class RenderPipeline(Component):
    def Start(self, params):
        self.Camera = params[0]

        self.RegisteredModels = []
        self.LoadedShaders = []

        self.ScreenWidth = 0
        self.ScreenHeight = 0

        self.LoadShaders()

        self.LightDir = Vector3Normalize(Vector3(0.35, -1.0, -0.35))
        self.SunColor = Vector3(253.0 / 255.0, 255.0 / 255.0, 232.0 / 255.0)
        self.SunStrength = 0.25
        self.SkyColor = Vector3(174.0 / 255.0, 183.0 / 255.0, 190.0 / 255.0)
        self.ShadowLight = ShadowLight()
        self.ShadowLight.target = Vector3(0.0, 0.0, 0.0)
        self.ShadowLight.position = Vector3Scale(self.LightDir, -20.0)
        self.ShadowLight.up = Vector3(0.0, 1.0, 0.0)
        self.ShadowLight.width = 20.0
        self.ShadowLight.height = 20.0
        self.ShadowLight.near = 1.0
        self.ShadowLight.far = 30.0
        self.ShadowMap = LoadShadowMap(2560, 1440)

        rlSetClipPlanes(0.01, 50.0)

    def RegisterModel(
        self,
        name,
        model,
        skinned_mesh,
        position=None,
        rotationAxis=None,
        rotationAngle=0.0,
        scale=None,
        color=RAYWHITE,
    ):
        if self.HasModel(model):
            print(f"Model {model} is already registered, skipping")
            return None

        registered = RegisteredModel(
            name=name,
            model=model,
            skinned_mesh=skinned_mesh,
            position=Vector3(0.0, 0.0, 0.0) if position is None else position,
            rotationAxis=(
                Vector3(0.0, 1.0, 0.0) if rotationAxis is None else rotationAxis
            ),
            rotationAngle=0.0 if rotationAngle is None else rotationAngle,
            scale=Vector3(1.0, 1.0, 1.0) if scale is None else scale,
            color=color,
        )

        self.RegisteredModels.append(registered)
        print(f"Registered model: {name} (skinned={skinned_mesh is not None})")
        return registered

    def UnregisterModel(self, model):
        for m in self.RegisteredModels:
            if m.model == model:
                self.RegisteredModels.remove(m)
                print(f"Unregistered model: {m.name}")
                return True
        print(f"Model not found for unregistration: {model.name}")

    def HasModel(self, model):
        return any(m.model == model for m in self.RegisteredModels)

    def SyncWindowSize(self):
        width, height = int(GetScreenWidth()), int(GetScreenHeight())
        if width != self.ScreenWidth or height != self.ScreenHeight:
            print(f"Render pipeline resolution set to {width}x{height}")
            if self.ScreenWidth > 0 and self.ScreenHeight > 0:
                # Unload old textures
                UnloadGBuffer(self.Gbuffer)
                UnloadRenderTexture(self.Lighted)
                UnloadRenderTexture(self.SsaoFront)
                UnloadRenderTexture(self.SsaoBack)
            # Set resolution
            self.ScreenWidth = width
            self.ScreenHeight = height
            # Recreate textures with new dimensions
            self.Gbuffer = LoadGBuffer(self.ScreenWidth, self.ScreenHeight)
            self.Lighted = LoadRenderTexture(self.ScreenWidth, self.ScreenHeight)
            self.SsaoFront = LoadRenderTexture(self.ScreenWidth, self.ScreenHeight)
            self.SsaoBack = LoadRenderTexture(self.ScreenWidth, self.ScreenHeight)

    def Render(self, debug):
        for registered in self.RegisteredModels:
            if registered.skinned_mesh:
                registered.skinned_mesh.Update()

        self.SyncWindowSize()

        self.ClipNearPtr = ffi.new("float*")
        self.ClipNearPtr[0] = rlGetCullDistanceNear()
        self.ClipFarPtr = ffi.new("float*")
        self.ClipFarPtr[0] = rlGetCullDistanceFar()

        self.RenderShadowMap()
        self.RenderGBuffer(0.5, 10.0)
        self.RenderSSAOShadows()
        self.RenderBlur()
        self.RenderLight()

        self.RenderBloom()
        self.RenderDebug(debug)
        self.RenderFXAA()

    def RenderShadowMap(self):
        # Render Shadow Maps
        BeginShadowMap(self.ShadowMap, self.ShadowLight)
        self.LightViewProj = MatrixMultiply(
            rlGetMatrixModelview(), rlGetMatrixProjection()
        )
        SetShaderValue(
            self.ShadowShader,
            self.ShadowShaderLightClipNear,
            self.ClipNearPtr,
            SHADER_UNIFORM_FLOAT,
        )
        SetShaderValue(
            self.ShadowShader,
            self.ShadowShaderLightClipFar,
            self.ClipFarPtr,
            SHADER_UNIFORM_FLOAT,
        )
        SetShaderValue(
            self.SkinnedShadowShader,
            self.SkinnedShadowShaderLightClipNear,
            self.ClipNearPtr,
            SHADER_UNIFORM_FLOAT,
        )
        SetShaderValue(
            self.SkinnedShadowShader,
            self.SkinnedShadowShaderLightClipFar,
            self.ClipFarPtr,
            SHADER_UNIFORM_FLOAT,
        )
        for registered in self.RegisteredModels:
            registered.Draw(
                self.SkinnedShadowShader
                if registered.skinned_mesh
                else self.ShadowShader
            )
        EndShadowMap()

    def RenderGBuffer(self, specularity, glossiness):
        # Render GBuffer
        BeginGBuffer(self.Gbuffer, self.Camera)
        self.CamView = rlGetMatrixModelview()
        self.CamProj = rlGetMatrixProjection()
        self.CamInvProj = MatrixInvert(self.CamProj)
        self.CamInvProjView = MatrixInvert(MatrixMultiply(self.CamView, self.CamProj))
        specularityPtr = ffi.new("float*")
        specularityPtr[0] = specularity
        glossinessPtr = ffi.new("float*")
        glossinessPtr[0] = glossiness
        SetShaderValue(
            self.BasicShader,
            self.BasicShaderSpecularity,
            specularityPtr,
            SHADER_UNIFORM_FLOAT,
        )
        SetShaderValue(
            self.BasicShader,
            self.BasicShaderGlossiness,
            glossinessPtr,
            SHADER_UNIFORM_FLOAT,
        )
        SetShaderValue(
            self.BasicShader,
            self.BasicShaderCamClipNear,
            self.ClipNearPtr,
            SHADER_UNIFORM_FLOAT,
        )
        SetShaderValue(
            self.BasicShader,
            self.BasicShaderCamClipFar,
            self.ClipFarPtr,
            SHADER_UNIFORM_FLOAT,
        )
        SetShaderValue(
            self.GridShader,
            self.GridShaderSpecularity,
            specularityPtr,
            SHADER_UNIFORM_FLOAT,
        )
        SetShaderValue(
            self.GridShader,
            self.GridShaderGlossiness,
            glossinessPtr,
            SHADER_UNIFORM_FLOAT,
        )
        SetShaderValue(
            self.GridShader,
            self.GridShaderCamClipNear,
            self.ClipNearPtr,
            SHADER_UNIFORM_FLOAT,
        )
        SetShaderValue(
            self.GridShader,
            self.GridShaderCamClipFar,
            self.ClipFarPtr,
            SHADER_UNIFORM_FLOAT,
        )
        SetShaderValue(
            self.SkinnedBasicShader,
            self.SkinnedBasicShaderSpecularity,
            specularityPtr,
            SHADER_UNIFORM_FLOAT,
        )
        SetShaderValue(
            self.SkinnedBasicShader,
            self.SkinnedBasicShaderGlossiness,
            glossinessPtr,
            SHADER_UNIFORM_FLOAT,
        )
        SetShaderValue(
            self.SkinnedBasicShader,
            self.SkinnedBasicShaderCamClipNear,
            self.ClipNearPtr,
            SHADER_UNIFORM_FLOAT,
        )
        SetShaderValue(
            self.SkinnedBasicShader,
            self.SkinnedBasicShaderCamClipFar,
            self.ClipFarPtr,
            SHADER_UNIFORM_FLOAT,
        )
        for registered in self.RegisteredModels:
            registered.Draw(
                self.SkinnedBasicShader if registered.skinned_mesh else self.GridShader
            )
        EndGBuffer(self.ScreenWidth, self.ScreenHeight)

    def RenderSSAOShadows(self):
        # Render SSAO and Shadows
        BeginTextureMode(self.SsaoFront)
        BeginShaderMode(self.SsaoShader)
        SetShaderValueTexture(
            self.SsaoShader, self.SsaoShaderGBufferNormal, self.Gbuffer.normal
        )
        SetShaderValueTexture(
            self.SsaoShader, self.SsaoShaderGBufferDepth, self.Gbuffer.depth
        )
        SetShaderValueMatrix(self.SsaoShader, self.SsaoShaderCamView, self.CamView)
        SetShaderValueMatrix(self.SsaoShader, self.SsaoShaderCamProj, self.CamProj)
        SetShaderValueMatrix(
            self.SsaoShader, self.SsaoShaderCamInvProj, self.CamInvProj
        )
        SetShaderValueMatrix(
            self.SsaoShader, self.SsaoShaderCamInvViewProj, self.CamInvProjView
        )
        SetShaderValueMatrix(
            self.SsaoShader, self.SsaoShaderLightViewProj, self.LightViewProj
        )
        SetShaderValueShadowMap(
            self.SsaoShader, self.SsaoShaderShadowMap, self.ShadowMap
        )
        SetShaderValue(
            self.SsaoShader,
            self.SsaoShaderShadowInvResolution,
            ffi.addressof(
                Vector2(
                    1.0 / self.ShadowMap.texture.width,
                    1.0 / self.ShadowMap.texture.height,
                )
            ),
            SHADER_UNIFORM_VEC2,
        )
        SetShaderValue(
            self.SsaoShader,
            self.SsaoShaderCamClipNear,
            self.ClipNearPtr,
            SHADER_UNIFORM_FLOAT,
        )
        SetShaderValue(
            self.SsaoShader,
            self.SsaoShaderCamClipFar,
            self.ClipFarPtr,
            SHADER_UNIFORM_FLOAT,
        )
        SetShaderValue(
            self.SsaoShader,
            self.SsaoShaderLightClipNear,
            self.ClipNearPtr,
            SHADER_UNIFORM_FLOAT,
        )
        SetShaderValue(
            self.SsaoShader,
            self.SsaoShaderLightClipFar,
            self.ClipFarPtr,
            SHADER_UNIFORM_FLOAT,
        )
        SetShaderValue(
            self.SsaoShader,
            self.SsaoShaderLightDir,
            ffi.addressof(self.LightDir),
            SHADER_UNIFORM_VEC3,
        )
        ClearBackground(WHITE)
        DrawTextureRec(
            self.SsaoFront.texture,
            Rectangle(
                0, 0, self.SsaoFront.texture.width, -self.SsaoFront.texture.height
            ),
            Vector2(0.0, 0.0),
            WHITE,
        )
        EndShaderMode()
        EndTextureMode()

    def RenderBlur(self):
        # Blur Horizontal
        BeginTextureMode(self.SsaoBack)
        BeginShaderMode(self.BlurShader)
        SetShaderValueTexture(
            self.BlurShader, self.BlurShaderGBufferNormal, self.Gbuffer.normal
        )
        SetShaderValueTexture(
            self.BlurShader, self.BlurShaderGBufferDepth, self.Gbuffer.depth
        )
        SetShaderValueTexture(
            self.BlurShader, self.BlurShaderInputTexture, self.SsaoFront.texture
        )
        SetShaderValueMatrix(
            self.BlurShader, self.BlurShaderCamInvProj, self.CamInvProj
        )
        SetShaderValue(
            self.BlurShader,
            self.BlurShaderCamClipNear,
            self.ClipNearPtr,
            SHADER_UNIFORM_FLOAT,
        )
        SetShaderValue(
            self.BlurShader,
            self.BlurShaderCamClipFar,
            self.ClipFarPtr,
            SHADER_UNIFORM_FLOAT,
        )
        SetShaderValue(
            self.BlurShader,
            self.BlurShaderInvTextureResolution,
            ffi.addressof(
                Vector2(
                    1.0 / self.SsaoFront.texture.width,
                    1.0 / self.SsaoFront.texture.height,
                )
            ),
            SHADER_UNIFORM_VEC2,
        )
        SetShaderValue(
            self.BlurShader,
            self.BlurShaderBlurDirection,
            ffi.addressof(Vector2(1.0, 0.0)),
            SHADER_UNIFORM_VEC2,
        )
        DrawTextureRec(
            self.SsaoBack.texture,
            Rectangle(0, 0, self.SsaoBack.texture.width, -self.SsaoBack.texture.height),
            Vector2(0, 0),
            WHITE,
        )
        EndShaderMode()
        EndTextureMode()

        # Blur Vertical
        BeginTextureMode(self.SsaoFront)
        BeginShaderMode(self.BlurShader)
        SetShaderValueTexture(
            self.BlurShader, self.BlurShaderInputTexture, self.SsaoBack.texture
        )
        SetShaderValue(
            self.BlurShader,
            self.BlurShaderBlurDirection,
            ffi.addressof(Vector2(0.0, 1.0)),
            SHADER_UNIFORM_VEC2,
        )
        DrawTextureRec(
            self.SsaoFront.texture,
            Rectangle(
                0, 0, self.SsaoFront.texture.width, -self.SsaoFront.texture.height
            ),
            Vector2(0, 0),
            WHITE,
        )
        EndShaderMode()
        EndTextureMode()

    def RenderLight(self):
        BeginTextureMode(self.Lighted)
        BeginShaderMode(self.LightingShader)
        sunStrengthPtr = ffi.new("float*")
        sunStrengthPtr[0] = self.SunStrength
        skyStrengthPtr = ffi.new("float*")
        skyStrengthPtr[0] = 0.15
        groundStrengthPtr = ffi.new("float*")
        groundStrengthPtr[0] = 0.1
        ambientStrengthPtr = ffi.new("float*")
        ambientStrengthPtr[0] = 1.0
        exposurePtr = ffi.new("float*")
        exposurePtr[0] = 0.9
        SetShaderValueTexture(
            self.LightingShader, self.LightingShaderGBufferColor, self.Gbuffer.color
        )
        SetShaderValueTexture(
            self.LightingShader, self.LightingShaderGBufferNormal, self.Gbuffer.normal
        )
        SetShaderValueTexture(
            self.LightingShader, self.LightingShaderGBufferDepth, self.Gbuffer.depth
        )
        SetShaderValueTexture(
            self.LightingShader, self.LightingShaderSSAO, self.SsaoFront.texture
        )
        SetShaderValue(
            self.LightingShader,
            self.LightingShaderCamPos,
            ffi.addressof(self.Camera.position),
            SHADER_UNIFORM_VEC3,
        )
        SetShaderValueMatrix(
            self.LightingShader, self.LightingShaderCamInvViewProj, self.CamInvProjView
        )
        SetShaderValue(
            self.LightingShader,
            self.LightingShaderLightDir,
            ffi.addressof(self.LightDir),
            SHADER_UNIFORM_VEC3,
        )
        SetShaderValue(
            self.LightingShader,
            self.LightingShaderSunColor,
            ffi.addressof(self.SunColor),
            SHADER_UNIFORM_VEC3,
        )
        SetShaderValue(
            self.LightingShader,
            self.LightingShaderSunStrength,
            sunStrengthPtr,
            SHADER_UNIFORM_FLOAT,
        )
        SetShaderValue(
            self.LightingShader,
            self.LightingShaderSkyColor,
            ffi.addressof(self.SkyColor),
            SHADER_UNIFORM_VEC3,
        )
        SetShaderValue(
            self.LightingShader,
            self.LightingShaderSkyStrength,
            skyStrengthPtr,
            SHADER_UNIFORM_FLOAT,
        )
        SetShaderValue(
            self.LightingShader,
            self.LightingShaderGroundStrength,
            groundStrengthPtr,
            SHADER_UNIFORM_FLOAT,
        )
        SetShaderValue(
            self.LightingShader,
            self.LightingShaderAmbientStrength,
            ambientStrengthPtr,
            SHADER_UNIFORM_FLOAT,
        )
        SetShaderValue(
            self.LightingShader,
            self.LightingShaderExposure,
            exposurePtr,
            SHADER_UNIFORM_FLOAT,
        )
        SetShaderValue(
            self.LightingShader,
            self.LightingShaderCamClipNear,
            self.ClipNearPtr,
            SHADER_UNIFORM_FLOAT,
        )
        SetShaderValue(
            self.LightingShader,
            self.LightingShaderCamClipFar,
            self.ClipFarPtr,
            SHADER_UNIFORM_FLOAT,
        )
        ClearBackground(RAYWHITE)
        DrawTextureRec(
            self.Gbuffer.color,
            Rectangle(0, 0, self.Gbuffer.color.width, -self.Gbuffer.color.height),
            Vector2(0, 0),
            WHITE,
        )
        EndShaderMode()

    def RenderBloom(self):
        BeginShaderMode(self.BloomShader)
        DrawTextureRec(
            self.Lighted.texture,
            Rectangle(0, 0, self.Lighted.texture.width, -self.Lighted.texture.height),
            Vector2(0, 0),
            WHITE,
        )
        EndShaderMode()

    def RenderDebug(self, debug):
        rlEnableColorBlend()
        BeginMode3D(self.Camera)
        debug()
        EndMode3D()
        rlDisableColorBlend()
        EndTextureMode()

    def RenderFXAA(self):
        BeginShaderMode(self.FxaaShader)
        SetShaderValueTexture(
            self.FxaaShader, self.FxaaShaderInputTexture, self.Lighted.texture
        )
        SetShaderValue(
            self.FxaaShader,
            self.FxaaShaderInvTextureResolution,
            ffi.addressof(
                Vector2(
                    1.0 / self.Lighted.texture.width, 1.0 / self.Lighted.texture.height
                )
            ),
            SHADER_UNIFORM_VEC2,
        )
        DrawTextureRec(
            self.Lighted.texture,
            Rectangle(0, 0, self.Lighted.texture.width, -self.Lighted.texture.height),
            Vector2(0, 0),
            WHITE,
        )
        EndShaderMode()

    def UnloadAll(self):
        UnloadRenderTexture(self.Lighted)
        UnloadRenderTexture(self.SsaoFront)
        UnloadRenderTexture(self.SsaoBack)
        UnloadGBuffer(self.Gbuffer)
        UnloadShadowMap(self.ShadowMap)
        for shader in self.LoadedShaders:
            UnloadShader(shader)

    def LoadShaders(self):
        self.ShadowShader = LoadShader("shadow.vs", "shadow.fs")
        self.ShadowShaderLightClipNear = GetShaderLocation(
            self.ShadowShader, b"lightClipNear"
        )
        self.ShadowShaderLightClipFar = GetShaderLocation(
            self.ShadowShader, b"lightClipFar"
        )
        self.LoadedShaders.append(self.ShadowShader)

        self.SkinnedShadowShader = LoadShader("skinnedShadow.vs", "shadow.fs")
        self.SkinnedShadowShaderLightClipNear = GetShaderLocation(
            self.SkinnedShadowShader, b"lightClipNear"
        )
        self.SkinnedShadowShaderLightClipFar = GetShaderLocation(
            self.SkinnedShadowShader, b"lightClipFar"
        )
        self.LoadedShaders.append(self.SkinnedShadowShader)

        self.SkinnedBasicShader = LoadShader("skinnedBasic.vs", "basic.fs")
        self.SkinnedBasicShaderSpecularity = GetShaderLocation(
            self.SkinnedBasicShader, b"specularity"
        )
        self.SkinnedBasicShaderGlossiness = GetShaderLocation(
            self.SkinnedBasicShader, b"glossiness"
        )
        self.SkinnedBasicShaderCamClipNear = GetShaderLocation(
            self.SkinnedBasicShader, b"camClipNear"
        )
        self.SkinnedBasicShaderCamClipFar = GetShaderLocation(
            self.SkinnedBasicShader, b"camClipFar"
        )
        self.LoadedShaders.append(self.SkinnedBasicShader)

        self.BasicShader = LoadShader("basic.vs", "basic.fs")
        self.BasicShaderSpecularity = GetShaderLocation(
            self.BasicShader, b"specularity"
        )
        self.BasicShaderGlossiness = GetShaderLocation(self.BasicShader, b"glossiness")
        self.BasicShaderCamClipNear = GetShaderLocation(
            self.BasicShader, b"camClipNear"
        )
        self.BasicShaderCamClipFar = GetShaderLocation(self.BasicShader, b"camClipFar")
        self.LoadedShaders.append(self.BasicShader)

        self.GridShader = LoadShader("grid.vs", "grid.fs")
        self.GridShaderSpecularity = GetShaderLocation(self.GridShader, b"specularity")
        self.GridShaderGlossiness = GetShaderLocation(self.GridShader, b"glossiness")
        self.GridShaderCamClipNear = GetShaderLocation(self.GridShader, b"camClipNear")
        self.GridShaderCamClipFar = GetShaderLocation(self.GridShader, b"camClipFar")
        self.LoadedShaders.append(self.GridShader)

        self.LightingShader = LoadShader("post.vs", "lighting.fs")
        self.LightingShaderGBufferColor = GetShaderLocation(
            self.LightingShader, b"gbufferColor"
        )
        self.LightingShaderGBufferNormal = GetShaderLocation(
            self.LightingShader, b"gbufferNormal"
        )
        self.LightingShaderGBufferDepth = GetShaderLocation(
            self.LightingShader, b"gbufferDepth"
        )
        self.LightingShaderSSAO = GetShaderLocation(self.LightingShader, b"ssao")
        self.LightingShaderCamPos = GetShaderLocation(self.LightingShader, b"camPos")
        self.LightingShaderCamInvViewProj = GetShaderLocation(
            self.LightingShader, b"camInvViewProj"
        )
        self.LightingShaderLightDir = GetShaderLocation(
            self.LightingShader, b"lightDir"
        )
        self.LightingShaderSunColor = GetShaderLocation(
            self.LightingShader, b"sunColor"
        )
        self.LightingShaderSunStrength = GetShaderLocation(
            self.LightingShader, b"sunStrength"
        )
        self.LightingShaderSkyColor = GetShaderLocation(
            self.LightingShader, b"skyColor"
        )
        self.LightingShaderSkyStrength = GetShaderLocation(
            self.LightingShader, b"skyStrength"
        )
        self.LightingShaderGroundStrength = GetShaderLocation(
            self.LightingShader, b"groundStrength"
        )
        self.LightingShaderAmbientStrength = GetShaderLocation(
            self.LightingShader, b"ambientStrength"
        )
        self.LightingShaderExposure = GetShaderLocation(
            self.LightingShader, b"exposure"
        )
        self.LightingShaderCamClipNear = GetShaderLocation(
            self.LightingShader, b"camClipNear"
        )
        self.LightingShaderCamClipFar = GetShaderLocation(
            self.LightingShader, b"camClipFar"
        )
        self.LoadedShaders.append(self.LightingShader)

        self.SsaoShader = LoadShader("post.vs", "ssao.fs")
        self.SsaoShaderGBufferNormal = GetShaderLocation(
            self.SsaoShader, b"gbufferNormal"
        )
        self.SsaoShaderGBufferDepth = GetShaderLocation(
            self.SsaoShader, b"gbufferDepth"
        )
        self.SsaoShaderCamView = GetShaderLocation(self.SsaoShader, b"camView")
        self.SsaoShaderCamProj = GetShaderLocation(self.SsaoShader, b"camProj")
        self.SsaoShaderCamInvProj = GetShaderLocation(self.SsaoShader, b"camInvProj")
        self.SsaoShaderCamInvViewProj = GetShaderLocation(
            self.SsaoShader, b"camInvViewProj"
        )
        self.SsaoShaderLightViewProj = GetShaderLocation(
            self.SsaoShader, b"lightViewProj"
        )
        self.SsaoShaderShadowMap = GetShaderLocation(self.SsaoShader, b"shadowMap")
        self.SsaoShaderShadowInvResolution = GetShaderLocation(
            self.SsaoShader, b"shadowInvResolution"
        )
        self.SsaoShaderCamClipNear = GetShaderLocation(self.SsaoShader, b"camClipNear")
        self.SsaoShaderCamClipFar = GetShaderLocation(self.SsaoShader, b"camClipFar")
        self.SsaoShaderLightClipNear = GetShaderLocation(
            self.SsaoShader, b"lightClipNear"
        )
        self.SsaoShaderLightClipFar = GetShaderLocation(
            self.SsaoShader, b"lightClipFar"
        )
        self.SsaoShaderLightDir = GetShaderLocation(self.SsaoShader, b"lightDir")
        self.LoadedShaders.append(self.SsaoShader)

        self.BlurShader = LoadShader("post.vs", "blur.fs")
        self.BlurShaderGBufferNormal = GetShaderLocation(
            self.BlurShader, b"gbufferNormal"
        )
        self.BlurShaderGBufferDepth = GetShaderLocation(
            self.BlurShader, b"gbufferDepth"
        )
        self.BlurShaderInputTexture = GetShaderLocation(
            self.BlurShader, b"inputTexture"
        )
        self.BlurShaderCamInvProj = GetShaderLocation(self.BlurShader, b"camInvProj")
        self.BlurShaderCamClipNear = GetShaderLocation(self.BlurShader, b"camClipNear")
        self.BlurShaderCamClipFar = GetShaderLocation(self.BlurShader, b"camClipFar")
        self.BlurShaderInvTextureResolution = GetShaderLocation(
            self.BlurShader, b"invTextureResolution"
        )
        self.BlurShaderBlurDirection = GetShaderLocation(
            self.BlurShader, b"blurDirection"
        )
        self.LoadedShaders.append(self.BlurShader)

        self.FxaaShader = LoadShader("post.vs", "fxaa.fs")
        self.FxaaShaderInputTexture = GetShaderLocation(
            self.FxaaShader, b"inputTexture"
        )
        self.FxaaShaderInvTextureResolution = GetShaderLocation(
            self.FxaaShader, b"invTextureResolution"
        )
        self.LoadedShaders.append(self.FxaaShader)

        self.BloomShader = LoadShader("post.vs", "bloom.fs")
        self.LoadedShaders.append(self.BloomShader)


# ----------------------------------------------------------------------------------
# Shadow Maps
# ----------------------------------------------------------------------------------
class ShadowLight:
    def __init__(self):
        self.target = Vector3Zero()
        self.position = Vector3Zero()
        self.up = Vector3(0.0, 1.0, 0.0)
        self.target = Vector3Zero()
        self.width = 0
        self.height = 0
        self.near = 0.0
        self.far = 1.0


def LoadShadowMap(width, height):
    target = RenderTexture()
    target.id = rlLoadFramebuffer()
    target.texture.width = width
    target.texture.height = height
    assert target.id != 0

    rlEnableFramebuffer(target.id)

    target.depth.id = rlLoadTextureDepth(width, height, False)
    target.depth.width = width
    target.depth.height = height
    target.depth.format = 19  # DEPTH_COMPONENT_24BIT?
    target.depth.mipmaps = 1
    rlFramebufferAttach(
        target.id, target.depth.id, RL_ATTACHMENT_DEPTH, RL_ATTACHMENT_TEXTURE2D, 0
    )
    assert rlFramebufferComplete(target.id)

    rlDisableFramebuffer()

    return target


def UnloadShadowMap(target):
    if target.id > 0:
        rlUnloadFramebuffer(target.id)


def BeginShadowMap(target, shadowLight):
    BeginTextureMode(target)
    ClearBackground(WHITE)

    rlDrawRenderBatchActive()  # Update and draw internal render batch

    rlMatrixMode(RL_PROJECTION)  # Switch to projection matrix
    rlPushMatrix()  # Save previous matrix, which contains the settings for the 2d ortho projection
    rlLoadIdentity()  # Reset current matrix (projection)

    rlOrtho(
        -shadowLight.width / 2,
        shadowLight.width / 2,
        -shadowLight.height / 2,
        shadowLight.height / 2,
        shadowLight.near,
        shadowLight.far,
    )

    rlMatrixMode(RL_MODELVIEW)  # Switch back to modelview matrix
    rlLoadIdentity()  # Reset current matrix (modelview)

    matView = MatrixLookAt(shadowLight.position, shadowLight.target, shadowLight.up)
    rlMultMatrixf(
        MatrixToFloatV(matView).v
    )  # Multiply modelview matrix by view matrix (camera)

    rlEnableDepthTest()  # Enable DEPTH_TEST for 3D


def EndShadowMap():
    rlDrawRenderBatchActive()  # Update and draw internal render batch

    rlMatrixMode(RL_PROJECTION)  # Switch to projection matrix
    rlPopMatrix()  # Restore previous matrix (projection) from matrix stack

    rlMatrixMode(RL_MODELVIEW)  # Switch back to modelview matrix
    rlLoadIdentity()  # Reset current matrix (modelview)

    rlDisableDepthTest()  # Disable DEPTH_TEST for 2D

    EndTextureMode()


def SetShaderValueShadowMap(shader, locIndex, target):
    if locIndex > -1:
        rlEnableShader(shader.id)
        slotPtr = ffi.new("int*")
        slotPtr[0] = 10  # Can be anything 0 to 15, but 0 will probably be taken up
        rlActiveTextureSlot(slotPtr[0])
        rlEnableTexture(target.depth.id)
        rlSetUniform(locIndex, slotPtr, SHADER_UNIFORM_INT, 1)


# ----------------------------------------------------------------------------------
# GBuffer
# ----------------------------------------------------------------------------------
class GBuffer:
    def __init__(self):
        self.id = 0  # OpenGL framebuffer object id
        self.color = Texture()  # Color buffer attachment texture
        self.normal = Texture()  # Normal buffer attachment texture
        self.depth = Texture()  # Depth buffer attachment texture


def LoadGBuffer(width, height):
    target = GBuffer()
    target.id = rlLoadFramebuffer()
    assert target.id

    rlEnableFramebuffer(target.id)

    target.color.id = rlLoadTexture(
        ffi.NULL, width, height, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8, 1
    )
    target.color.width = width
    target.color.height = height
    target.color.format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8
    target.color.mipmaps = 1
    rlFramebufferAttach(
        target.id,
        target.color.id,
        RL_ATTACHMENT_COLOR_CHANNEL0,
        RL_ATTACHMENT_TEXTURE2D,
        0,
    )

    target.normal.id = rlLoadTexture(
        ffi.NULL, width, height, PIXELFORMAT_UNCOMPRESSED_R16G16B16A16, 1
    )
    target.normal.width = width
    target.normal.height = height
    target.normal.format = PIXELFORMAT_UNCOMPRESSED_R16G16B16A16
    target.normal.mipmaps = 1
    rlFramebufferAttach(
        target.id,
        target.normal.id,
        RL_ATTACHMENT_COLOR_CHANNEL1,
        RL_ATTACHMENT_TEXTURE2D,
        0,
    )

    target.depth.id = rlLoadTextureDepth(width, height, False)
    target.depth.width = width
    target.depth.height = height
    target.depth.format = 19  # DEPTH_COMPONENT_24BIT?
    target.depth.mipmaps = 1
    rlFramebufferAttach(
        target.id, target.depth.id, RL_ATTACHMENT_DEPTH, RL_ATTACHMENT_TEXTURE2D, 0
    )

    assert rlFramebufferComplete(target.id)

    rlDisableFramebuffer()

    return target


def UnloadGBuffer(target):
    if target.id > 0:
        rlUnloadFramebuffer(target.id)


def BeginGBuffer(target, camera):
    rlDrawRenderBatchActive()  # Update and draw internal render batch

    rlEnableFramebuffer(target.id)  # Enable render target
    rlActiveDrawBuffers(2)

    # Set viewport and RLGL internal framebuffer size
    rlViewport(0, 0, target.color.width, target.color.height)
    rlSetFramebufferWidth(target.color.width)
    rlSetFramebufferHeight(target.color.height)

    ClearBackground(BLACK)

    rlMatrixMode(RL_PROJECTION)  # Switch to projection matrix
    rlPushMatrix()  # Save previous matrix, which contains the settings for the 2d ortho projection
    rlLoadIdentity()  # Reset current matrix (projection)

    aspect = float(target.color.width) / float(target.color.height)

    # NOTE: zNear and zFar values are important when computing depth buffer values
    if camera.projection == CAMERA_PERSPECTIVE:
        top = rlGetCullDistanceNear() * np.tan(camera.fovy * 0.5 * DEG2RAD)
        right = top * aspect
        rlFrustum(
            -right, right, -top, top, rlGetCullDistanceNear(), rlGetCullDistanceFar()
        )

    elif camera.projection == CAMERA_ORTHOGRAPHIC:
        top = camera.fovy / 2.0
        right = top * aspect
        rlOrtho(
            -right, right, -top, top, rlGetCullDistanceNear(), rlGetCullDistanceFar()
        )

    rlMatrixMode(RL_MODELVIEW)  # Switch back to modelview matrix
    rlLoadIdentity()  # Reset current matrix (modelview)

    # Setup Camera view
    matView = MatrixLookAt(camera.position, camera.target, camera.up)
    rlMultMatrixf(
        MatrixToFloatV(matView).v
    )  # Multiply modelview matrix by view matrix (camera)

    rlEnableDepthTest()  # Enable DEPTH_TEST for 3D


def EndGBuffer(windowWidth, windowHeight):
    rlDrawRenderBatchActive()  # Update and draw internal render batch

    rlDisableDepthTest()  # Disable DEPTH_TEST for 2D
    rlActiveDrawBuffers(1)
    rlDisableFramebuffer()  # Disable render target (fbo)

    rlMatrixMode(RL_PROJECTION)  # Switch to projection matrix
    rlPopMatrix()  # Restore previous matrix (projection) from matrix stack
    rlLoadIdentity()  # Reset current matrix (projection)
    rlOrtho(0, windowWidth, windowHeight, 0, 0.0, 1.0)

    rlMatrixMode(RL_MODELVIEW)  # Switch back to modelview matrix
    rlLoadIdentity()  # Reset current matrix (modelview)
