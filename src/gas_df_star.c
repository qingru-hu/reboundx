/**
 * @file    gas_df_star.c
 * @brief   Gas drag from a thin, disk with a power-law density profile 
 * @author  Aleksey Generozov
 * 
 * 
 *
 *
 * This file is part of reboundx.
 *
 * reboundx is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * reboundx is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with rebound.  If not, see <http://www.gnu.org/licenses/>.
 *
 * The section after the dollar signs gets built into the documentation by a script.  All lines must start with space * space like below.
 * Tables always must be preceded and followed by a blank line.  See http://docutils.sourceforge.net/docs/user/rst/quickstart.html for a primer on rst.
 * $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
 *
 * $Gas Effects$       // Effect category (must be the first non-blank line after dollar signs and between dollar signs to be detected by script).
 *
 * ======================= ===============================================
 * Authors                 A. Generozov, H. Perets
 * Implementation Paper    `Generozov and Perets 2022 <https://arxiv.org/abs/2212.11301>`_
 * Based on                `Ostriker 1999 (with simplifications) <https://ui.adsabs.harvard.edu/abs/1999ApJ...513..252O/abstract>`_, `Just et al 2012 <https://ui.adsabs.harvard.edu/abs/2012ApJ...758...51J/abstract>`_.
 * C Example               :ref:`c_example_gas_dynamical_friction`
 * Python Example          `GasDynamicalFriction.ipynb <https://github.com/dtamayo/reboundx/blob/master/ipython_examples/GasDynamicalFriction.ipynb>`_
 *                        
 * 
 * ======================= ===============================================
 * 
 * 
 * **Effect Parameters**
 * 
 * ============================ =========== ==================================================================
 * Field (C type)               Required    Description
 * ============================ =========== ==================================================================
 * rhog (double)                Yes         Normalization of density. Density in the disk midplane is rhog*r^alpha_rhog
 * alpha_rhog (double)          Yes         Power-law slope of the power-law density profile.
 * cs (double)                  Yes         Normalization of the sound speed. Sound speed has profile cs*r^alpha_cs
 * alpha_cs (double)            Yes         Power-law slope of the sound speed
 * xmin (double)                Yes         Dimensionless parameter that determines the Coulomb logarithm (ln(L) =log (1/xmin))
 * hr (double)                  Yes         Aspect ratio of the disk
 * Qd (double)                  Yes         Prefactor for geometric drag
 * ============================ =========== ==================================================================
 * 
 * 
 * **Particle Parameters**
 * 
 * None.
 * 
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "reboundx.h"

#include <stdio.h>

static double mass_inclosed(const double r){
    //Mass enclosed within radius r in Rsun
    //Using MESA fit for 1Msun RG model
    const double a=32.85;
    const double b=0.52;
    const double c=-3.31;
    const double d=29.79;
    const double mass_loc = (r>9.9)?62.44932716040428:(a * tanh(b * (r + c)) + d); // in Msun
    return mass_loc;
}


static double mach_piece_sub(const double mach){
    //Using powerlaw expansion at low mach numbers for numerical reasons
    if (mach<0.02){
        return mach*mach*mach/3.+mach*mach*mach*mach*mach/5.;
    }
    //Subsonic expression from Ostriker...
    return 0.5*log((1.0+mach)/(1.0-mach))-mach;

}


static double calculate_pre_factor(const double mach, const double vel, const double t,
    const double xmin){
    //Simplified version of the Ostriker dynamical friction formula...
    const double coul=log(1.0/xmin);
    if (mach>=1.0){
        return coul;
    }
    else{
        return fmin(coul, mach_piece_sub(mach));
    }

}


static void rebx_calculate_gas_df_star(struct reb_simulation* const sim, struct reb_particle* const particles, const int N, const double xmin, const double xcenter, const double ycenter, const double zcenter, const double df_switch, const double grav_switch){ 
    const int _N_real = sim->N - sim->N_var;
    // 1. 计算当前双星系统的质心 (COM)
    // 使用 REBOUND 内置函数获取整个系统的质心位置和速度
    const struct reb_particle com = reb_simulation_com(sim);
#pragma omp parallel for
    for (int i=0;i<_N_real;i++){
        const struct reb_particle p = particles[i];
        // 假设气体球心始终跟随双星质心
        const double r_rel[3] = {p.x - com.x - xcenter, p.y - com.y - ycenter, p.z - com.z - zcenter};
        const double v_rel[3] = {p.vx - com.vx, p.vy - com.vy, p.vz - com.vz};
        // const double r_rel[3] = {p.x, p.y, p.z};
        // const double v_rel[3] = {p.vx, p.vy, p.vz};
        // 3. 计算相对速度的大小
        const double r_mag = sqrt(r_rel[0]*r_rel[0] + r_rel[1]*r_rel[1] + r_rel[2]*r_rel[2]);
        const double v_mag = sqrt(v_rel[0]*v_rel[0] + v_rel[1]*v_rel[1] + v_rel[2]*v_rel[2]);

        const double au2rsun = 215.03215567054764; // in rsun
        double r_in_rsun = r_mag * au2rsun;
        const double cs_coeff[6] = {
            -2.09949012e-02, 
            1.91774068e-01, 
            2.94736358e+00, 
            -3.91610718e+01, 
            1.30103395e+01, 
            1.10150833e+03
        };
        // 替代 cs_loc 的计算
        double cs_loc = ((((
            cs_coeff[0]*r_in_rsun + 
            cs_coeff[1])*r_in_rsun + 
            cs_coeff[2])*r_in_rsun + 
            cs_coeff[3])*r_in_rsun + 
            cs_coeff[4])*r_in_rsun + 
            cs_coeff[5]; // in km/s
        cs_loc *= 0.210945; // in au/yr

        const double rho_coeff[6] = {
            3.18430772e-04, 
           -9.50213464e-03, 
            1.00418290e-01, 
           -4.00488083e-01, 
            5.05114296e-02, 
            2.31333938e+00
        };
        double rho_loc = (r_in_rsun<(9.9))?(((((
        rho_coeff[0] * r_in_rsun + 
        rho_coeff[1]) * r_in_rsun + 
        rho_coeff[2]) * r_in_rsun + 
        rho_coeff[3]) * r_in_rsun + 
        rho_coeff[4]) * r_in_rsun + 
        rho_coeff[5]):0; // in g/cm3
        rho_loc *= 1683721.7643842339; // in Msun/au**3

        const double mach=v_mag/cs_loc;
        const double t=sim->t;

        const double integ=calculate_pre_factor(mach, v_mag, t, xmin);

        
        const double mp = p.m;
        const double v_soft = 1e-3; // 或根據系統自定義
        const double r_soft = 1e-3; // 或根據系統自定義
        const double fc=4.*M_PI*(sim->G*sim->G)*mp*(rho_loc)/(v_mag*v_mag*v_mag + v_soft)*integ;
        const double fg= - sim->G * mass_inclosed(r_in_rsun) / (r_mag*r_mag*r_mag + r_soft);

        // particles[i].ax = particles[i].ax + fg*r_rel[0] - fc*v_rel[0];
        // particles[i].ay = particles[i].ay + fg*r_rel[1] - fc*v_rel[1];
        // particles[i].az = particles[i].az + fg*r_rel[2] - fc*v_rel[2];
        particles[i].ax += fg*r_rel[0]*grav_switch;
        particles[i].ay += fg*r_rel[1]*grav_switch;
        particles[i].az += fg*r_rel[2]*grav_switch;

        particles[i].ax -= fc*v_rel[0]*df_switch;
        particles[i].ay -= fc*v_rel[1]*df_switch;
        particles[i].az -= fc*v_rel[2]*df_switch;
    }
}

void rebx_gas_df_star(struct reb_simulation* const sim, struct rebx_force* const force,\
    struct reb_particle* const particles, const int N){
    struct rebx_extras* const rebx = sim->extras;
    double* xcenter= rebx_get_param(rebx, force->ap, "gas_df_xcen");
    if (xcenter == NULL){
        reb_simulation_error(sim, "Need to specify the x coordinate of the star center \n");
    }
    double* ycenter= rebx_get_param(rebx, force->ap, "gas_df_ycen");
    if (ycenter == NULL){
        reb_simulation_error(sim, "Need to specify the y coordinate of the star center \n");
    }
    double* zcenter= rebx_get_param(rebx, force->ap, "gas_df_zcen");
    if (zcenter == NULL){
        reb_simulation_error(sim, "Need to specify the z coordinate of the star center \n");
    }
    double* xmin= rebx_get_param(rebx, force->ap, "gas_df_xmin");
    if (xmin == NULL){
        reb_simulation_error(sim, "Need to set a cutoff.\n");
    }
    double* df_switch= rebx_get_param(rebx, force->ap, "gas_df_switch");
    if (df_switch == NULL){
        reb_simulation_error(sim, "Need to decide whether include the DF.\n");
    }
    double* grav_switch= rebx_get_param(rebx, force->ap, "gas_grav_switch");
    if (grav_switch == NULL){
        reb_simulation_error(sim, "Need to decide whether include the DF.\n");
    }
    rebx_calculate_gas_df_star(sim, particles, N, *xmin, *xcenter, *ycenter, *zcenter, *df_switch, *grav_switch);

}


/*
int main(){
    double testm = mass_inclosed(6.957e10);
    printf("%f\n", testm);
}
*/