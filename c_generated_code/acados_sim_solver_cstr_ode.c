/*
 * Copyright (c) The acados authors.
 *
 * This file is part of acados.
 *
 * The 2-Clause BSD License
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.;
 */
// standard
#include <stdio.h>
#include <stdlib.h>

// acados
#include "acados_c/external_function_interface.h"
#include "acados_c/sim_interface.h"
#include "acados_c/external_function_interface.h"

#include "acados/sim/sim_common.h"
#include "acados/utils/external_function_generic.h"
#include "acados/utils/print.h"


// example specific
#include "cstr_ode_model/cstr_ode_model.h"
#include "acados_sim_solver_cstr_ode.h"


// ** solver data **

cstr_ode_sim_solver_capsule * cstr_ode_acados_sim_solver_create_capsule()
{
    void* capsule_mem = malloc(sizeof(cstr_ode_sim_solver_capsule));
    cstr_ode_sim_solver_capsule *capsule = (cstr_ode_sim_solver_capsule *) capsule_mem;

    return capsule;
}


int cstr_ode_acados_sim_solver_free_capsule(cstr_ode_sim_solver_capsule * capsule)
{
    free(capsule);
    return 0;
}


int cstr_ode_acados_sim_create(cstr_ode_sim_solver_capsule * capsule)
{
    // initialize
    const int nx = CSTR_ODE_NX;
    const int nu = CSTR_ODE_NU;
    const int nz = CSTR_ODE_NZ;
    const int np = CSTR_ODE_NP;
    bool tmp_bool;

    
    double Tsim = 0.25;

    

    // sim plan & config
    sim_solver_plan_t plan;
    plan.sim_solver = DISCRETE;

    // create correct config based on plan
    sim_config * cstr_ode_sim_config = sim_config_create(plan);
    capsule->acados_sim_config = cstr_ode_sim_config;

    // sim dims
    void *cstr_ode_sim_dims = sim_dims_create(cstr_ode_sim_config);
    capsule->acados_sim_dims = cstr_ode_sim_dims;
    sim_dims_set(cstr_ode_sim_config, cstr_ode_sim_dims, "nx", &nx);
    sim_dims_set(cstr_ode_sim_config, cstr_ode_sim_dims, "nu", &nu);
    sim_dims_set(cstr_ode_sim_config, cstr_ode_sim_dims, "nz", &nz);


    // sim opts
    sim_opts *cstr_ode_sim_opts = sim_opts_create(cstr_ode_sim_config, cstr_ode_sim_dims);
    capsule->acados_sim_opts = cstr_ode_sim_opts;
    int tmp_int = 3;
    sim_opts_set(cstr_ode_sim_config, cstr_ode_sim_opts, "newton_iter", &tmp_int);
    double tmp_double = 0;
    sim_opts_set(cstr_ode_sim_config, cstr_ode_sim_opts, "newton_tol", &tmp_double);
    sim_collocation_type collocation_type = GAUSS_LEGENDRE;
    sim_opts_set(cstr_ode_sim_config, cstr_ode_sim_opts, "collocation_type", &collocation_type);

 
    tmp_int = 4;
    sim_opts_set(cstr_ode_sim_config, cstr_ode_sim_opts, "num_stages", &tmp_int);
    tmp_int = 1;
    sim_opts_set(cstr_ode_sim_config, cstr_ode_sim_opts, "num_steps", &tmp_int);
    tmp_bool = 0;
    sim_opts_set(cstr_ode_sim_config, cstr_ode_sim_opts, "jac_reuse", &tmp_bool);


    // sim in / out
    sim_in *cstr_ode_sim_in = sim_in_create(cstr_ode_sim_config, cstr_ode_sim_dims);
    capsule->acados_sim_in = cstr_ode_sim_in;
    sim_out *cstr_ode_sim_out = sim_out_create(cstr_ode_sim_config, cstr_ode_sim_dims);
    capsule->acados_sim_out = cstr_ode_sim_out;

    sim_in_set(cstr_ode_sim_config, cstr_ode_sim_dims,
               cstr_ode_sim_in, "T", &Tsim);

    // model functions

    // sim solver
    sim_solver *cstr_ode_sim_solver = sim_solver_create(cstr_ode_sim_config,
                                               cstr_ode_sim_dims, cstr_ode_sim_opts);
    capsule->acados_sim_solver = cstr_ode_sim_solver;


    /* initialize parameter values */
    double* p = calloc(np, sizeof(double));
    
    p[0] = 0.1;

    cstr_ode_acados_sim_update_params(capsule, p, np);
    free(p);


    /* initialize input */
    // x
    double x0[3];
    for (int ii = 0; ii < 3; ii++)
        x0[ii] = 0.0;

    sim_in_set(cstr_ode_sim_config, cstr_ode_sim_dims,
               cstr_ode_sim_in, "x", x0);


    // u
    double u0[2];
    for (int ii = 0; ii < 2; ii++)
        u0[ii] = 0.0;

    sim_in_set(cstr_ode_sim_config, cstr_ode_sim_dims,
               cstr_ode_sim_in, "u", u0);

    // S_forw
    double S_forw[15];
    for (int ii = 0; ii < 15; ii++)
        S_forw[ii] = 0.0;
    for (int ii = 0; ii < 3; ii++)
        S_forw[ii + ii * 3 ] = 1.0;


    sim_in_set(cstr_ode_sim_config, cstr_ode_sim_dims,
               cstr_ode_sim_in, "S_forw", S_forw);

    int status = sim_precompute(cstr_ode_sim_solver, cstr_ode_sim_in, cstr_ode_sim_out);

    return status;
}


int cstr_ode_acados_sim_solve(cstr_ode_sim_solver_capsule *capsule)
{
    // integrate dynamics using acados sim_solver
    int status = sim_solve(capsule->acados_sim_solver,
                           capsule->acados_sim_in, capsule->acados_sim_out);
    if (status != 0)
        printf("error in cstr_ode_acados_sim_solve()! Exiting.\n");

    return status;
}


int cstr_ode_acados_sim_free(cstr_ode_sim_solver_capsule *capsule)
{
    // free memory
    sim_solver_destroy(capsule->acados_sim_solver);
    sim_in_destroy(capsule->acados_sim_in);
    sim_out_destroy(capsule->acados_sim_out);
    sim_opts_destroy(capsule->acados_sim_opts);
    sim_dims_destroy(capsule->acados_sim_dims);
    sim_config_destroy(capsule->acados_sim_config);

    // free external function

    return 0;
}


int cstr_ode_acados_sim_update_params(cstr_ode_sim_solver_capsule *capsule, double *p, int np)
{
    int status = 0;
    int casadi_np = CSTR_ODE_NP;

    if (casadi_np != np) {
        printf("cstr_ode_acados_sim_update_params: trying to set %i parameters for external functions."
            " External function has %i parameters. Exiting.\n", np, casadi_np);
        exit(1);
    }

    return status;
}

/* getters pointers to C objects*/
sim_config * cstr_ode_acados_get_sim_config(cstr_ode_sim_solver_capsule *capsule)
{
    return capsule->acados_sim_config;
};

sim_in * cstr_ode_acados_get_sim_in(cstr_ode_sim_solver_capsule *capsule)
{
    return capsule->acados_sim_in;
};

sim_out * cstr_ode_acados_get_sim_out(cstr_ode_sim_solver_capsule *capsule)
{
    return capsule->acados_sim_out;
};

void * cstr_ode_acados_get_sim_dims(cstr_ode_sim_solver_capsule *capsule)
{
    return capsule->acados_sim_dims;
};

sim_opts * cstr_ode_acados_get_sim_opts(cstr_ode_sim_solver_capsule *capsule)
{
    return capsule->acados_sim_opts;
};

sim_solver  * cstr_ode_acados_get_sim_solver(cstr_ode_sim_solver_capsule *capsule)
{
    return capsule->acados_sim_solver;
};

