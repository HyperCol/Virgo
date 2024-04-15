#version 130

#define Reflection_Temporal_Upsample

#ifdef Reflection_Temporal_Upsample
    #define Reflection_Render_Scale 0.5
    #define Stage_Scale Reflection_Render_Scale
#endif

#include "/program/post.vsh"