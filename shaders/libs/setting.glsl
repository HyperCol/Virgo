//Mod Support
#define DH_Support

//Antialiasing
#define Enabled_Temporal_AA
    #define TAA_Sampling_Sharpeness 70          //removed [1 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100]
    #define TAA_Post_Processing_Sharpeness 35   //[1 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100]
    #define TAA_Post_Sharpen_Limit 0.002        //[0.001 0.0015 0.002 0.0025 0.003]
    //#define TAA_Non_Clip

//Materials
#define Parallax_Mapping
//#define Voxel_Parallax_Mapping
#define Parallax_Mapping_Depth_Low_Detail OFF   //[OFF Default 16 32 64 96 128 160 192 256]

//Lighting
#define GI_A_Trous_4
#define GI_A_Trous_3
#define GI_A_Trous_2
#define GI_A_Trous_1

//Camera Setting
#define Enabled_Bloom
    #define Bloom_Exporuse 3.5         //[1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0 5.5 6.0]

#define White_Balance_Adjustment OFF    //[OFF 100 1000 2000 3000 4000 5000 6000 6500 7000 8000 9000 10000 11000 12000 13000]

#define Camera_FPS 24                   //[8 12 24 30 40 60 90 120 144]
#define Film_Grain 5                    //[OFF 0 1 2 3 4 5 6 7 8 9 10]

#define Camera_ISO 50                  //[25 50 75 100 125 150 175 200 300 400 600 800]
#define Camera_Auto_Exposure
#define Camera_Exporsure_Value 0.0      //[-4.0 -3.5 -3.0 -2.5 -2.0 -1.5 -1.0 -0.75 -0.5 -0.25 0.0 0.25 0.5 0.75 1.0 1.5 2.0 2.5 3.0 3.5 4.0]
#define Camera_Auto_Min_EV -3.0         //[-4.0 -3.5 -3.0 -2.5 -2.0 -1.5 -1.0 -0.5 0.0]
#define Camera_Auto_Max_EV 2.0          //[0.0 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0]

#define Camera_Aperture 2.8             //[1.0 1.4 2.0 2.8 4.0 5.6 8.0 11.0 16.0 22.0 32.0 44.0 64.0]
#define Camera_Focal_Length 0.004       //[0.001 0.002 0.003 0.004 0.005 0.006 0.007 0.008 0.009 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1]

//Lighting Setting
#define Torch_Light_Color_R 255
#define Torch_Light_Color_G 210
#define Torch_Light_Color_B 165
#define Torch_Light_Temperature 4000        //[1000 1500 2000 2500 3000 3500 4000 4500 5000 5500 6000 6500]

#define Sun_Light_Luminance 1.0
#define Moon_Light_Luminance 0.05

//#define SS_Contact_Shadow_Tracing_Clip

//#define Nature_Light_Exposure 0.0       //[-2.0 -1.5 -1.0 -0.5 0.0 0.5 1.0 1.5 2.0]
#define Sky_Texture_Exposure 0.0        //[-2.0 -1.5 -1.0 -0.5 0.0 0.5 1.0 1.5 2.0]
#define Emission_Texture_Exposure 0.5   //[-5.0 -4.0 -3.0 -2.0 -1.0 0.0 1.0 2.0 3.0 4.0 5.0]
#define Emissive_Light_Exposure 2.0     //[-3.0 -2.0 -1.0 0.0 1.0 2.0 3.0]
#define Shadow_Light_Exposure 1.25       //[1.0 1.54 1.73 2.0]

#define Diffuse_Accumulation_Frame 20   //[0 5 10 15 20 30 40 50 60]

#define World_Scale 10.0                //[1.0 2.5 5.0 7.5 10.0 12.5 15.0 17.5 20.0 30.0 40.0 50.0]
#define Altitude_Scale 1.0              //[1.0 2.5 5.0 7.5 10.0 12.5 15.0 17.5 20.0 30.0 40.0 50.0]
#define Altitude_Start 300.0

#define Fog_Light_Extinction_Distance 1000.0
#define High_Density_Fog_Distance_Limit 100.0
#define Rain_Fog_Density 0.00         //
#define Biome_Fog_Density 0.000001      //default 0.0 or 1/100 rain density
#define Fog_Front_Scattering_Weight 0.1
#define Fog_FrontScattering_Phase 0.8
#define Fog_BackScattering_Phase 0.35
    #define Fog_Moon_Light_Phase_Multiplier 0.6666

#define Enabled_Lower_Clouds
#define Lower_Clouds_Scattering 0.08    //[0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12]
#define Lower_Clouds_Distance OFF       //[OFF 10000 15000 20000 25000 30000 35000 40000 45000 50000] not for framerate
#define Lower_Clouds_Bottom 1000.0
#define Lower_Clouds_Top 1500.0
#define Lower_Clouds_Quality 24
#define Lower_Clouds_Light_Quality 8
    #define Lower_Clouds_Light_Tracing_Max_Distance 1200.0

//#define Enabled_Medium_Clouds
#define Medium_Clouds_Bottom 4000.0
#define Medium_Clouds_Top 4400.0
#define Medium_Clouds_Scattering 0.03    //[0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12]


//
#define Brightness_Bar 102
#define OFF 0
#define Low 1
#define Medium 2
#define High 3
#define Ultra 4

#ifdef GI_A_Trous_4
#endif

#ifdef GI_A_Trous_3
#endif

#ifdef GI_A_Trous_2
#endif
 
#ifdef GI_A_Trous_1
#endif

#ifdef Lower_Clouds_Distance
#endif

//Atmospheric Scattering
//#define rayleigh_absorption