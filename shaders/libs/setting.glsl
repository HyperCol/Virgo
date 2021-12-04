//Option
#define Auto -2
#define OFF -1
#define Low 0
#define Medium 1
#define High 2
#define Ultra 3

#define PCSS 0
#define Vaule 1

#define RGB 0
#define Color_Temperature 1

//Lighting Setting
#define Sun_Light_Luminance 6.0
#define Moon_Light_Luminance 0.1
#define Blocks_Light_Luminance 0.2
#define Nature_Light_Min_Luminance 0.1

//Blocks Light
#define Held_Light_Quality Medium                   //[Medium High]
#define Blocks_Light_Color Color_Temperature        //[RGB Color_Temperature]
#define Blocks_Light_Color_Temperture 2500.0        //[1700.0 1850.0 2000.0 2500.0 3000.0 3200.0 3275.0 3380.0 5000.0 5500.0 6000.0 6500.0 7000.0 8000.0]
#define Blocks_Light_Color_R 1.0                    //[0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0]
#define Blocks_Light_Color_G 0.7                    //[0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0]
#define Blocks_Light_Color_B 0.3                    //[0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0]
#define Blocks_Light_Intensity 1.0                  //[0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0]

//Sun Light
#define Soft_Shadow_Quality High                    //[OFF High Ultra]
#define Soft_Shadow_Penumbra PCSS                   //[PCSS 1 2 4 8 16 32]

#define Enabled_Screen_Space_Contact_Shadow         //

//Ambient Light
#define SSAO_Quality Medium                         //[OFF Medium] HBAO quality
    #define SSAO_Falloff 0.7                        //
    #define SSAO_Bias 0.0                           //
    #define SSAO_Direction_Step 4                   //
    #define SSAO_Rotation_Step 8                    //
    #define SSAO_Radius 0.3                         //
    #define SSAO_Low_Resolution                     //half resolution render ssao

//#define Disabled_Sky_Occlusion
#define Sky_Light_Level_Min 0                       //[0 1 2 3 4 5 6 7 8 9 10 11 12 13 14]

//Materials
#define Small_SlimeBlock_Density 8.0                //[0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0]
#define Small_HoneyBlock_Density 8.0                //[0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0]

#define Translucent_Blocks_Quality High             //[Low High]

//Atmosphere
#define Land_Atmospheric_Scattering_Quality Low     //[Low Medium High Ultra]
#define Land_Atmospheric_Density 10.0               //[1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 20.0 30.0 40.0 50.0 60.0 70.0 80.0 90.0 100.0]
#define Land_Atmospheric_Distribution 100.0         //[1.0 5.0 10.0 50.0 100.0 500.0 1000.0]

#define Sky_Atmospheric_Scattering_Quality High     //[Low Medium High]

#define Atmospheric_Shape Sphere                    //[Sphere Cube]

//#define Soft_Planet_Shadow                          //

#define Planet_Radius 6360000.0                     //[]
#define Atmosphere_Radius 6420000.0                 //[]

#define Custom -1
#define Earth_Like 0
#define Atmosphere_Profile Earth_Like               //[Custom Earth_Like]

#define Rayleigh_Scattering 1.0                     //[0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0]
#define Rayleigh_Absorption 0.0                     //[0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0]
#define Rayleigh_Transmittance_R 4.0                //[1.0 2.0 4.0 6.0 8.0 10.0 12.0 14.0 16.0 18.0 20.0 22.0 24.0 26.0 28.0 30.0 32.0 34.0]
#define Rayleigh_Transmittance_G 12.0               //[1.0 2.0 4.0 6.0 8.0 10.0 12.0 14.0 16.0 18.0 20.0 22.0 24.0 26.0 28.0 30.0 32.0 34.0]
#define Rayleigh_Transmittance_B 32.0               //[1.0 2.0 4.0 6.0 8.0 10.0 12.0 14.0 16.0 18.0 20.0 22.0 24.0 26.0 28.0 30.0 32.0 34.0]
#define Rayleigh_Distribution 8000.0                //[500.0 1000.0 1500.0 2000.0 2500.0 3000.0 3500.0 4000.0 4500.0 5000.0 5500.0 6000.0 6500.0 7000.0 7500.0 8000.0 8500.0 9000.0 9500.0 10000.0]

#define Mie_Scattering 1.0                          //[0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0]
#define Mie_Absorption 0.1                          //[0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0]
#define Mie_Transmittance_R 4.0                     //[1.0 2.0 4.0 6.0 8.0 10.0 12.0 14.0 16.0 18.0 20.0 22.0 24.0 26.0 28.0 30.0 32.0 34.0]
#define Mie_Transmittance_G 4.0                     //[1.0 2.0 4.0 6.0 8.0 10.0 12.0 14.0 16.0 18.0 20.0 22.0 24.0 26.0 28.0 30.0 32.0 34.0]
#define Mie_Transmittance_B 4.0                     //[1.0 2.0 4.0 6.0 8.0 10.0 12.0 14.0 16.0 18.0 20.0 22.0 24.0 26.0 28.0 30.0 32.0 34.0]
#define Mie_Distribution 1000.0                     //[500.0 1000.0 1500.0 2000.0 2500.0 3000.0 3500.0 4000.0 4500.0 5000.0 5500.0 6000.0 6500.0 7000.0 7500.0 8000.0 8500.0 9000.0 9500.0 10000.0]

#define Ozone_Scattering 0.0                        //[0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0]
#define Ozone_Absorption 1.0                        //[0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0]
#define Ozone_Transmittance_R 0.41112               //[]
#define Ozone_Transmittance_G 0.99576               //[]
#define Ozone_Transmittance_B 0.00427               //[]
#define Ozone_Height_Height 25000.0                 //[]
#define Ozone_Height_Thickness 15000.0              //[]

//Sky
#define Moon_Texture_Luminance 4.0                  //[1.0 2.0 3.0 4.0 5.0 6.0 7.0]
#define Moon_Radius 1.0                             //[0.125 0.25 0.5 1.0 2.0 4.0 8.0]
#define Moon_Distance 1.0                           //[0.125 0.25 0.5 1.0 2.0 4.0 8.0]

#define Stars_Fade_In 0.1                           //[0.01 0.05 0.1 0.15 0.2]
#define Stars_Fade_Out 0.0                          //[-1.0 -0.25 -0.2 -0.15 -0.1 -0.05 0.0]
#define Stars_Visible 0.005                         //[0.00062 0.00125 0.0025 0.005 0.01 0.02 0.04]
#define Stars_Luminance 0.1                         //[0.1 0.25 0.5 0.75 1.0 2.5 5.0 7.5 10.0 25.0 50.0 75.0 100.0]
#define Stars_Speed 1.0                             //[0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0]
#define Planet_Angle 0.1                            //[-0.3 -0.2 -0.1 0.0 0.1 0.2 0.3 0.4 0.5] +:north -:south

#define Polaris_Size 2.0                            //[1.0 2.0 3.0 4.0]
#define Polaris_Luminance 1.0                       //[1.0]
#define Polaris_Offset_X 4.0                        //[1.0 2.0 3.0 4.0 5.0 6.0 7.0]
#define Polaris_Offset_Y 4.0                        //[1.0 2.0 3.0 4.0 5.0 6.0 7.0]

//Camera Settings
#define Camera_ISO 200                              //[25 50 100 200 400 800 1600]

#define Camera_Exposure_Value 0.0                   //[-4.0 -3.0 -2.0 -1.0 0.0 1.0 2.0 3.0 4.0]
#define Camera_Exposure_Min_EV -2.0                 //[-5.0 -4.0 -3.5 -3.0 -2.5 -2.0 -1.5 -1.0 -0.5 0.0 1.0]
#define Camera_Exposure_Max_EV 6.0                  //[3.0 4.0 4.5 5.0 5.5 6.0 6.5 7.0 7.5 8.0 9.0]

#define Camera_Average_Exposure                     //
#define Camera_Exposure_Delay 2.0                   //[0.5 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0]

//#define Camera_DOF                                  //
#define Camera_Aperture 2.8                         //[1.0 1.4 2.0 2.8 4.0 5.6 8.0 11.0 16.0 22.0 32.0 44.0 64.0 6400.0]
#define Camera_Focal_Length 0.004                   //[0.001 0.002 0.003 0.004 0.005 0.006 0.007 0.008 0.009 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1]
#define Camera_Focal_Distance_Auto                  //
#define Camera_Focal_Distance 1.0                   //[0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 2.5 3.0 3.5 4.0 4.5 5.0 6.0 7.0 8.0 9.0 10.0]

#define Camera_FPS 45.0                             //[25.0 30.0 45.0 60.0 90.0 120.0 144.0 240.0]
#define Camera_Shutter_Speed 0                      //[0 10 20 30 40 50 60 70 80 90 100 150 200 300 400]

#define Enabled_Bloom
    //#define Bloom_Intensity 0.2                         //[0.05 0.1 0.2 0.4 0.6 0.8 1.0 2.0 4.0 8.0 16.0]
    #define Bloom_Exposure_Value -3.0                     //[-5.0 -4.5 -4.0 -3.5 -3.0 -2.5 -2.0 -1.5 -1.0]
    //#define Bloom_Intensity_Test                    //

//Antialiasing
#define Enabled_TAA
#define TAA_Accumulation_Shapress 50                //[0 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100]
#define TAA_Post_Processing_Sharpeness 50           //[0 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100]
#define TAA_Post_Processing_Sharpen_Limit 0.125     //[0.5 0.25 0.125 0.0625 0.03125]
//#define TAA_No_Clip