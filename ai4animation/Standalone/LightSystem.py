# Copyright (c) Meta Platforms, Inc. and affiliates.
from dataclasses import dataclass
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
    LoadShader,
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


class LightSystem(Component):
    def Start(self, params):
        self.Camera = params[0]

        self.RegisteredModels = []

        # Shaders
        self.ShadowShader = LoadShader(
            utils.ToBytes(
                utils.GetDirectory(__file__) + "./resources/shaders/shadow.vs"
            ),
            utils.ToBytes(
                utils.GetDirectory(__file__) + "./resources/shaders/shadow.fs"
            ),
        )
        self.ShadowShaderLightClipNear = GetShaderLocation(
            self.ShadowShader, b"lightClipNear"
        )
        self.ShadowShaderLightClipFar = GetShaderLocation(
            self.ShadowShader, b"lightClipFar"
        )

        self.SkinnedShadowShader = LoadShader(
            utils.ToBytes(
                utils.GetDirectory(__file__) + "./resources/shaders/skinnedShadow.vs"
            ),
            utils.ToBytes(
                utils.GetDirectory(__file__) + "./resources/shaders/shadow.fs"
            ),
        )
        self.SkinnedShadowShaderLightClipNear = GetShaderLocation(
            self.SkinnedShadowShader, b"lightClipNear"
        )
        self.SkinnedShadowShaderLightClipFar = GetShaderLocation(
            self.SkinnedShadowShader, b"lightClipFar"
        )

        self.SkinnedBasicShader = LoadShader(
            utils.ToBytes(
                utils.GetDirectory(__file__) + "./resources/shaders/skinnedBasic.vs"
            ),
            utils.ToBytes(
                utils.GetDirectory(__file__) + "./resources/shaders/basic.fs"
            ),
        )
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

        self.BasicShader = LoadShader(
            utils.ToBytes(
                utils.GetDirectory(__file__) + "./resources/shaders/basic.vs"
            ),
            utils.ToBytes(
                utils.GetDirectory(__file__) + "./resources/shaders/basic.fs"
            ),
        )
        self.BasicShaderSpecularity = GetShaderLocation(
            self.BasicShader, b"specularity"
        )
        self.BasicShaderGlossiness = GetShaderLocation(self.BasicShader, b"glossiness")
        self.BasicShaderCamClipNear = GetShaderLocation(
            self.BasicShader, b"camClipNear"
        )
        self.BasicShaderCamClipFar = GetShaderLocation(self.BasicShader, b"camClipFar")

        self.GridShader = LoadShader(
            utils.ToBytes(utils.GetDirectory(__file__) + "./resources/shaders/grid.vs"),
            utils.ToBytes(utils.GetDirectory(__file__) + "./resources/shaders/grid.fs"),
        )
        self.GridShaderSpecularity = GetShaderLocation(self.GridShader, b"specularity")
        self.GridShaderGlossiness = GetShaderLocation(self.GridShader, b"glossiness")
        self.GridShaderCamClipNear = GetShaderLocation(self.GridShader, b"camClipNear")
        self.GridShaderCamClipFar = GetShaderLocation(self.GridShader, b"camClipFar")

        self.LightingShader = LoadShader(
            utils.ToBytes(utils.GetDirectory(__file__) + "./resources/shaders/post.vs"),
            utils.ToBytes(
                utils.GetDirectory(__file__) + "./resources/shaders/lighting.fs"
            ),
        )
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

        self.SsaoShader = LoadShader(
            utils.ToBytes(utils.GetDirectory(__file__) + "./resources/shaders/post.vs"),
            utils.ToBytes(utils.GetDirectory(__file__) + "./resources/shaders/ssao.fs"),
        )
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

        self.BlurShader = LoadShader(
            utils.ToBytes(utils.GetDirectory(__file__) + "./resources/shaders/post.vs"),
            utils.ToBytes(utils.GetDirectory(__file__) + "./resources/shaders/blur.fs"),
        )
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

        self.FxaaShader = LoadShader(
            utils.ToBytes(utils.GetDirectory(__file__) + "./resources/shaders/post.vs"),
            utils.ToBytes(utils.GetDirectory(__file__) + "./resources/shaders/fxaa.fs"),
        )
        self.FxaaShaderInputTexture = GetShaderLocation(
            self.FxaaShader, b"inputTexture"
        )
        self.FxaaShaderInvTextureResolution = GetShaderLocation(
            self.FxaaShader, b"invTextureResolution"
        )

        self.BloomShader = LoadShader(
            utils.ToBytes(utils.GetDirectory(__file__) + "./resources/shaders/post.vs"),
            utils.ToBytes(
                utils.GetDirectory(__file__) + "./resources/shaders/bloom.fs"
            ),
        )

        rlSetClipPlanes(0.01, 50.0)

        # Shadows
        self.LightDir = Vector3Normalize(Vector3(0.35, -1.0, -0.35))

        self.ShadowLight = ShadowLight()
        self.ShadowLight.target = Vector3(0.0, 0.0, 0.0)
        self.ShadowLight.position = Vector3Scale(self.LightDir, -20.0)
        self.ShadowLight.up = Vector3(0.0, 1.0, 0.0)
        self.ShadowLight.width = 20.0
        self.ShadowLight.height = 20.0
        self.ShadowLight.near = 1.0
        self.ShadowLight.far = 30.0

        self.ShadowWidth = 4096
        self.ShadowHeight = 4096
        self.ShadowInvResolution = Vector2(
            1.0 / self.ShadowWidth, 1.0 / self.ShadowHeight
        )
        self.ShadowMap = LoadShadowMap(self.ShadowWidth, self.ShadowHeight)

        # Sun
        self.SunColor = Vector3(253.0 / 255.0, 255.0 / 255.0, 232.0 / 255.0)
        self.SunStrength = 0.25
        self.SkyColor = Vector3(174.0 / 255.0, 183.0 / 255.0, 190.0 / 255.0)

        # GBuffer and Render Textures
        self.ScreenWidth = int(GetScreenWidth())
        self.ScreenHeight = int(GetScreenHeight())
        self.Gbuffer = LoadGBuffer(self.ScreenWidth, self.ScreenHeight)
        self.Lighted = LoadRenderTexture(self.ScreenWidth, self.ScreenHeight)
        self.SsaoFront = LoadRenderTexture(self.ScreenWidth, self.ScreenHeight)
        self.SsaoBack = LoadRenderTexture(self.ScreenWidth, self.ScreenHeight)

    def RegisterModel(
        self,
        name,
        model,
        skinned_mesh,
        position=None,
        rotationAxis=None,
        rotationAngle=0.0,
        scale=None,
        color=WHITE,
    ):
        if self.HasModel(model):
            print(f"Model {model} is already registered, skipping")
            return None

        registered = RegisteredModel(
            name=name,
            model=model,
            skinned_mesh=skinned_mesh,
            position=Vector3(0.0, 0.0, 0.0) if position is None else position,
            rotationAxis=Vector3(0.0, 1.0, 0.0)
            if rotationAxis is None
            else rotationAxis,
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

    def HandleWindowResize(self):
        """Check if window has been resized and recreate render textures if needed"""
        currentWidth = int(GetScreenWidth())
        currentHeight = int(GetScreenHeight())

        if currentWidth != self.ScreenWidth or currentHeight != self.ScreenHeight:
            print(
                f"Window resized from {self.ScreenWidth}x{self.ScreenHeight} to {currentWidth}x{currentHeight}"
            )

            # Unload old textures
            UnloadRenderTexture(self.Lighted)
            UnloadRenderTexture(self.SsaoBack)
            UnloadRenderTexture(self.SsaoFront)
            UnloadGBuffer(self.Gbuffer)

            # Update dimensions
            self.ScreenWidth = currentWidth
            self.ScreenHeight = currentHeight

            # Recreate textures with new dimensions
            self.Gbuffer = LoadGBuffer(self.ScreenWidth, self.ScreenHeight)
            self.Lighted = LoadRenderTexture(self.ScreenWidth, self.ScreenHeight)
            self.SsaoFront = LoadRenderTexture(self.ScreenWidth, self.ScreenHeight)
            self.SsaoBack = LoadRenderTexture(self.ScreenWidth, self.ScreenHeight)

    def Render(self, draw):
        for registered in self.RegisteredModels:
            if registered.skinned_mesh:
                registered.skinned_mesh.Update()

        # Handle window resize
        self.HandleWindowResize()

        # Render Shadow Maps
        BeginShadowMap(self.ShadowMap, self.ShadowLight)

        lightViewProj = MatrixMultiply(rlGetMatrixModelview(), rlGetMatrixProjection())
        lightClipNear = rlGetCullDistanceNear()
        lightClipFar = rlGetCullDistanceFar()

        lightClipNearPtr = ffi.new("float*")
        lightClipNearPtr[0] = lightClipNear
        lightClipFarPtr = ffi.new("float*")
        lightClipFarPtr[0] = lightClipFar

        SetShaderValue(
            self.ShadowShader,
            self.ShadowShaderLightClipNear,
            lightClipNearPtr,
            SHADER_UNIFORM_FLOAT,
        )
        SetShaderValue(
            self.ShadowShader,
            self.ShadowShaderLightClipFar,
            lightClipFarPtr,
            SHADER_UNIFORM_FLOAT,
        )
        SetShaderValue(
            self.SkinnedShadowShader,
            self.SkinnedShadowShaderLightClipNear,
            lightClipNearPtr,
            SHADER_UNIFORM_FLOAT,
        )
        SetShaderValue(
            self.SkinnedShadowShader,
            self.SkinnedShadowShaderLightClipFar,
            lightClipFarPtr,
            SHADER_UNIFORM_FLOAT,
        )

        for registered in self.RegisteredModels:
            registered.Draw(
                self.SkinnedShadowShader
                if registered.skinned_mesh
                else self.ShadowShader
            )

        EndShadowMap()

        # Render GBuffer
        BeginGBuffer(self.Gbuffer, self.Camera)

        camView = rlGetMatrixModelview()
        camProj = rlGetMatrixProjection()
        camInvProj = MatrixInvert(camProj)
        camInvViewProj = MatrixInvert(MatrixMultiply(camView, camProj))
        camClipNear = rlGetCullDistanceNear()
        camClipFar = rlGetCullDistanceFar()

        camClipNearPtr = ffi.new("float*")
        camClipNearPtr[0] = camClipNear
        camClipFarPtr = ffi.new("float*")
        camClipFarPtr[0] = camClipFar

        specularityPtr = ffi.new("float*")
        specularityPtr[0] = 0.5
        glossinessPtr = ffi.new("float*")
        glossinessPtr[0] = 10.0

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
            camClipNearPtr,
            SHADER_UNIFORM_FLOAT,
        )
        SetShaderValue(
            self.BasicShader,
            self.BasicShaderCamClipFar,
            camClipFarPtr,
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
            camClipNearPtr,
            SHADER_UNIFORM_FLOAT,
        )
        SetShaderValue(
            self.GridShader,
            self.GridShaderCamClipFar,
            camClipFarPtr,
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
            camClipNearPtr,
            SHADER_UNIFORM_FLOAT,
        )
        SetShaderValue(
            self.SkinnedBasicShader,
            self.SkinnedBasicShaderCamClipFar,
            camClipFarPtr,
            SHADER_UNIFORM_FLOAT,
        )

        for registered in self.RegisteredModels:
            registered.Draw(
                self.SkinnedBasicShader if registered.skinned_mesh else self.GridShader
            )

        EndGBuffer(self.ScreenWidth, self.ScreenHeight)

        # Render SSAO and Shadows
        BeginTextureMode(self.SsaoFront)

        BeginShaderMode(self.SsaoShader)

        SetShaderValueTexture(
            self.SsaoShader, self.SsaoShaderGBufferNormal, self.Gbuffer.normal
        )
        SetShaderValueTexture(
            self.SsaoShader, self.SsaoShaderGBufferDepth, self.Gbuffer.depth
        )
        SetShaderValueMatrix(self.SsaoShader, self.SsaoShaderCamView, camView)
        SetShaderValueMatrix(self.SsaoShader, self.SsaoShaderCamProj, camProj)
        SetShaderValueMatrix(self.SsaoShader, self.SsaoShaderCamInvProj, camInvProj)
        SetShaderValueMatrix(
            self.SsaoShader, self.SsaoShaderCamInvViewProj, camInvViewProj
        )
        SetShaderValueMatrix(
            self.SsaoShader, self.SsaoShaderLightViewProj, lightViewProj
        )
        SetShaderValueShadowMap(
            self.SsaoShader, self.SsaoShaderShadowMap, self.ShadowMap
        )
        SetShaderValue(
            self.SsaoShader,
            self.SsaoShaderShadowInvResolution,
            ffi.addressof(self.ShadowInvResolution),
            SHADER_UNIFORM_VEC2,
        )
        SetShaderValue(
            self.SsaoShader,
            self.SsaoShaderCamClipNear,
            camClipNearPtr,
            SHADER_UNIFORM_FLOAT,
        )
        SetShaderValue(
            self.SsaoShader,
            self.SsaoShaderCamClipFar,
            camClipFarPtr,
            SHADER_UNIFORM_FLOAT,
        )
        SetShaderValue(
            self.SsaoShader,
            self.SsaoShaderLightClipNear,
            lightClipNearPtr,
            SHADER_UNIFORM_FLOAT,
        )
        SetShaderValue(
            self.SsaoShader,
            self.SsaoShaderLightClipFar,
            lightClipFarPtr,
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

        # Blur Horizontal
        BeginTextureMode(self.SsaoBack)

        BeginShaderMode(self.BlurShader)

        blurDirection = Vector2(1.0, 0.0)
        blurInvTextureResolution = Vector2(
            1.0 / self.SsaoFront.texture.width, 1.0 / self.SsaoFront.texture.height
        )

        SetShaderValueTexture(
            self.BlurShader, self.BlurShaderGBufferNormal, self.Gbuffer.normal
        )
        SetShaderValueTexture(
            self.BlurShader, self.BlurShaderGBufferDepth, self.Gbuffer.depth
        )
        SetShaderValueTexture(
            self.BlurShader, self.BlurShaderInputTexture, self.SsaoFront.texture
        )
        SetShaderValueMatrix(self.BlurShader, self.BlurShaderCamInvProj, camInvProj)
        SetShaderValue(
            self.BlurShader,
            self.BlurShaderCamClipNear,
            camClipNearPtr,
            SHADER_UNIFORM_FLOAT,
        )
        SetShaderValue(
            self.BlurShader,
            self.BlurShaderCamClipFar,
            camClipFarPtr,
            SHADER_UNIFORM_FLOAT,
        )
        SetShaderValue(
            self.BlurShader,
            self.BlurShaderInvTextureResolution,
            ffi.addressof(blurInvTextureResolution),
            SHADER_UNIFORM_VEC2,
        )
        SetShaderValue(
            self.BlurShader,
            self.BlurShaderBlurDirection,
            ffi.addressof(blurDirection),
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

        blurDirection = Vector2(0.0, 1.0)

        SetShaderValueTexture(
            self.BlurShader, self.BlurShaderInputTexture, self.SsaoBack.texture
        )
        SetShaderValue(
            self.BlurShader,
            self.BlurShaderBlurDirection,
            ffi.addressof(blurDirection),
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

        # Light GBuffer
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
            self.LightingShader, self.LightingShaderCamInvViewProj, camInvViewProj
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
            camClipNearPtr,
            SHADER_UNIFORM_FLOAT,
        )
        SetShaderValue(
            self.LightingShader,
            self.LightingShaderCamClipFar,
            camClipFarPtr,
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

        # # Render Bloom

        # BeginShaderMode(self.BloomShader)

        # # fxaaInvTextureResolution = Vector2(1.0 / self.Lighted.texture.width, 1.0 / self.Lighted.texture.height)

        # # SetShaderValueTexture(self.FxaaShader, self.FxaaShaderInputTexture, self.Lighted.texture)
        # # SetShaderValue(self.FxaaShader, self.FxaaShaderInvTextureResolution, ffi.addressof(fxaaInvTextureResolution), SHADER_UNIFORM_VEC2)

        # DrawTextureRec(
        #     self.Lighted.texture,
        #     Rectangle(0, 0, self.Lighted.texture.width, -self.Lighted.texture.height),
        #     Vector2(0, 0),
        #     WHITE)

        # EndShaderMode()

        # Debug Draw
        rlEnableColorBlend()
        BeginMode3D(self.Camera)
        draw()
        EndMode3D()
        rlDisableColorBlend()

        EndTextureMode()

        # Render Final with FXAA

        BeginShaderMode(self.FxaaShader)

        fxaaInvTextureResolution = Vector2(
            1.0 / self.Lighted.texture.width, 1.0 / self.Lighted.texture.height
        )

        SetShaderValueTexture(
            self.FxaaShader, self.FxaaShaderInputTexture, self.Lighted.texture
        )
        SetShaderValue(
            self.FxaaShader,
            self.FxaaShaderInvTextureResolution,
            ffi.addressof(fxaaInvTextureResolution),
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
        UnloadRenderTexture(self.SsaoBack)
        UnloadRenderTexture(self.SsaoFront)
        UnloadGBuffer(self.Gbuffer)

        UnloadShadowMap(self.ShadowMap)

        # TODO: Unload registered Models

        UnloadShader(self.FxaaShader)
        UnloadShader(self.BlurShader)
        UnloadShader(self.SsaoShader)
        UnloadShader(self.LightingShader)
        UnloadShader(self.BasicShader)
        UnloadShader(self.GridShader)
        UnloadShader(self.SkinnedBasicShader)
        UnloadShader(self.SkinnedShadowShader)
        UnloadShader(self.ShadowShader)


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

    # Setup Camera view
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
        # Setup perspective projection
        top = rlGetCullDistanceNear() * np.tan(camera.fovy * 0.5 * DEG2RAD)
        right = top * aspect

        rlFrustum(
            -right, right, -top, top, rlGetCullDistanceNear(), rlGetCullDistanceFar()
        )

    elif camera.projection == CAMERA_ORTHOGRAPHIC:
        # Setup orthographic projection
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
