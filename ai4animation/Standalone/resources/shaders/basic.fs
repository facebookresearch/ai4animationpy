#version 300 es
precision highp float;

in vec3 fragPosition;
in vec2 fragTexCoord;
in vec4 fragColor;
in vec3 fragNormal;

uniform vec4 colDiffuse;
uniform float specularity;
uniform float glossiness;
uniform float camClipNear;
uniform float camClipFar;

layout (location = 0) out vec4 gbufferColor;
layout (location = 1) out vec4 gbufferNormal;

vec3 FromGamma(in vec3 col)
{
    return vec3(pow(col.x, 1.0/2.2), pow(col.y, 1.0/2.2), pow(col.z, 1.0/2.2));
}

float LinearDepth(float depth, float near, float far)
{
    return (2.0 * near) / (far + near - depth * (far - near));
}

void main()
{
    vec3 albedo = FromGamma(fragColor.xyz * colDiffuse.xyz);
    float spec = specularity;

    gbufferColor = vec4(albedo, spec);
    gbufferNormal = vec4(fragNormal * 0.5f + 0.5f, glossiness / 100.0f);
    gl_FragDepth = LinearDepth(gl_FragCoord.z, camClipNear, camClipFar);
}
