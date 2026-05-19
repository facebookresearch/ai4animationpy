#version 410

in vec3 vertexPosition;
in vec4 vertexBoneIds;
in vec4 vertexBoneWeights;

#define MAX_BONE_NUM 128
uniform mat4 boneMatrices[MAX_BONE_NUM];

uniform mat4 mvp;
uniform vec3 lightDir;

void main()
{
    int boneIndex0 = int(vertexBoneIds.x);
    int boneIndex1 = int(vertexBoneIds.y);
    int boneIndex2 = int(vertexBoneIds.z);
    int boneIndex3 = int(vertexBoneIds.w);

    vec4 skinnedPosition =
        vertexBoneWeights.x * (boneMatrices[boneIndex0] * vec4(vertexPosition, 1.0)) +
        vertexBoneWeights.y * (boneMatrices[boneIndex1] * vec4(vertexPosition, 1.0)) +
        vertexBoneWeights.z * (boneMatrices[boneIndex2] * vec4(vertexPosition, 1.0)) +
        vertexBoneWeights.w * (boneMatrices[boneIndex3] * vec4(vertexPosition, 1.0));

    // worldPos = skinnedPosition (matModel is identity for skinned characters)
    vec3 worldPos = skinnedPosition.xyz / skinnedPosition.w;

    // Project onto Y=0 ground plane along light direction.
    // When light is nearly horizontal, projection goes to infinity — clamp t.
    float denom = max(-lightDir.y, 0.001);
    float t = clamp(worldPos.y / denom, 0.0, 50.0);
    vec3 shadowPos = worldPos.xyz + lightDir * t;
    shadowPos.y = 0.005;

    // mvp = viewProj when matModel is identity
    gl_Position = mvp * vec4(shadowPos, 1.0);
}
