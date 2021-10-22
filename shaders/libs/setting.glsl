//
#define OFF -1
#define Low 0
#define Medium 1
#define High 2
#define Ultra 3
//
#define Sun_Light_Luminance 8.0
#define Moon_Light_Luminance 0.1
#define Blocks_Light_Luminance 0.5

#define Held_Light_Quality Medium                   //[Medium High]

#define Atmospheric_Rayleigh_Scattering 1.0
#define Atmospheric_Rayleigh_Absorption 0.0
#define Atmospheric_Mie_Scattering 1.0
#define Atmospheric_Mie_Absorption 1.0
#define Atmospheric_Ozone_Scattering 0.0
#define Atmospheric_Ozone_Absorption 1.0
#define Atmospheric_Shape Sphere                    //[Sphere Cube]

#define Moon_Texture_Luminance 10.0     //[1.0 5.0 7.5 10.0 15.0 20.0 100.0]
#define Moon_Radius 1.0                 //[0.5 0.75 1.0 1.5 2.0]
#define Moon_Distance 1.0               //[0.5 0.75 1.0 1.5 2.0]

#define Stars_Visible 0.005         //[0.00062 0.00125 0.0025 0.005 0.01 0.02 0.04]
#define Stars_Luminance 0.1         //[0.025 0.05 0.1 0.2 0.4]
#define Stars_Speed 1.0             //[0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0]
#define Planet_Angle 0.1            //[-0.3 -0.2 -0.1 0.0 0.1 0.2 0.3 0.4 0.5] +:north -:south
#define Polaris_Size 2.0            //[1.0 2.0 3.0 4.0]
#define Polaris_Luminance 1.0       //[1.0]
#define Polaris_Offset_X 4.0        //[1.0 2.0 3.0 4.0 5.0 6.0 7.0]
#define Polaris_Offset_Y 4.0        //[1.0 2.0 3.0 4.0 5.0 6.0 7.0]

#define Enabled_TAA
#define TAA_Accumulation_Shapress 50                //[0 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100]
#define TAA_Post_Processing_Sharpeness 50           //[0 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100]
#define TAA_Post_Processing_Sharpen_Limit 0.125     //[0.5 0.25 0.125 0.0625 0.03125]