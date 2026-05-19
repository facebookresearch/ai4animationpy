#version 410

in vec3 fragNormal;
in vec4 fragColor;

uniform vec4 colDiffuse;

out vec4 finalColor;

void main()
{
    vec3 lightDir = normalize(vec3(0.35, -1.0, -0.35));
    vec3 normal = normalize(fragNormal);

    // Lambertian diffuse
    float diff = max(dot(normal, -lightDir), 0.0);

    // Hemisphere ambient (sky blue from above, ground brown from below)
    float up = normal.y * 0.5 + 0.5;
    vec3 ambient = mix(vec3(0.15, 0.12, 0.10), vec3(0.25, 0.28, 0.35), up);

    vec3 color = fragColor.rgb * colDiffuse.rgb;
    finalColor = vec4(color * (ambient + diff * 0.65), colDiffuse.a);
}
