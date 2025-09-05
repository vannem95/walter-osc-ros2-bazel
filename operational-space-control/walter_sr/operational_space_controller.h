#pragma once

#include <filesystem>
#include <vector>
#include <string>
#include <cstdlib>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <iostream>
#include <cassert>

#include <typeinfo>

#include "absl/status/status.h"
#include "absl/log/absl_check.h"

#include "mujoco/mujoco.h"
#include "Eigen/Dense"
#include "Eigen/SparseCore"
#include "osqp++.h"
#include "osqp.h"

#include "operational-space-control/walter_sr/utilities.h"
#include "operational-space-control/utilities.h"

#include "operational-space-control/walter_sr/autogen/autogen_functions.h"
#include "operational-space-control/walter_sr/autogen/autogen_defines.h"

#include "operational-space-control/walter_sr/aliases.h"
#include "operational-space-control/walter_sr/constants.h"
#include "operational-space-control/walter_sr/containers.h"


#include "rclcpp/rclcpp.hpp"



using namespace operational_space_controller::constants;
using namespace operational_space_controller::containers;
using namespace operational_space_controller::aliases;
using namespace osqp;

// Anonymous Namespace for shorthand constants:
namespace {
    // Map Casadi Functions to FunctionOperations Struct:
    FunctionOperations Aeq_ops{
        .incref=Aeq_incref,
        .checkout=Aeq_checkout,
        .eval=Aeq,
        .release=Aeq_release,
        .decref=Aeq_decref
    };

    FunctionOperations beq_ops{
        .incref=beq_incref,
        .checkout=beq_checkout,
        .eval=beq,
        .release=beq_release,
        .decref=beq_decref
    };

    FunctionOperations Aineq_ops{
        .incref=Aineq_incref,
        .checkout=Aineq_checkout,
        .eval=Aineq,
        .release=Aineq_release,
        .decref=Aineq_decref
    };

    FunctionOperations bineq_ops{
        .incref=bineq_incref,
        .checkout=bineq_checkout,
        .eval=bineq,
        .release=bineq_release,
        .decref=bineq_decref
    };

    FunctionOperations H_ops{
        .incref=H_incref,
        .checkout=H_checkout,
        .eval=H,
        .release=H_release,
        .decref=H_decref
    };

    FunctionOperations f_ops{
        .incref=f_incref,
        .checkout=f_checkout,
        .eval=f,
        .release=f_release,
        .decref=f_decref
    };

    // Casadi Functions
    using AeqParams = 
        FunctionParams<Aeq_SZ_ARG, Aeq_SZ_RES, Aeq_SZ_IW, Aeq_SZ_W, optimization::Aeq_rows, optimization::Aeq_cols, optimization::Aeq_sz, 4>;
    using beqParams =
        FunctionParams<beq_SZ_ARG, beq_SZ_RES, beq_SZ_IW, beq_SZ_W, optimization::beq_sz, 1, optimization::beq_sz, 4>;
    using AineqParams =
        FunctionParams<Aineq_SZ_ARG, Aineq_SZ_RES, Aineq_SZ_IW, Aineq_SZ_W, optimization::Aineq_rows, optimization::Aineq_cols, optimization::Aineq_sz, 1>;
    using bineqParams =
        FunctionParams<bineq_SZ_ARG, bineq_SZ_RES, bineq_SZ_IW, bineq_SZ_W, optimization::bineq_sz, 1, optimization::bineq_sz, 1>;
    using HParams =
        FunctionParams<H_SZ_ARG, H_SZ_RES, H_SZ_IW, H_SZ_W, optimization::H_rows, optimization::H_cols, optimization::H_sz, 4>;
    using fParams =
        FunctionParams<f_SZ_ARG, f_SZ_RES, f_SZ_IW, f_SZ_W, optimization::f_sz, 1, optimization::f_sz, 4>;
}

//TODO(jeh15): Refactor all voids with absl::Status
class OperationalSpaceController {
    public:
        OperationalSpaceController(std::filesystem::path xml_path, int control_rate_us = 2000, OsqpSettings osqp_settings = OsqpSettings()) : 
            xml_path(xml_path), control_rate_us(control_rate_us), settings(osqp_settings) {}
        ~OperationalSpaceController() {}

        absl::Status initialize(State initial_state) {
            char error[1000];
            mj_model = mj_loadXML(xml_path.c_str(), nullptr, error, 1000);
            if( !mj_model ) {
                printf("%s\n", error);
                return absl::InternalError("Failed to load Mujoco Model");
            }

            // Physics timestep:
            mj_model->opt.timestep = 0.002;
            
            mj_data = mj_makeData(mj_model);

            for(const std::string_view& site : model::site_list) {
                std::string site_str = std::string(site);
                int id = mj_name2id(mj_model, mjOBJ_SITE, site_str.data());
                assert(id != -1 && "Site not found in model.");
                sites.push_back(site_str);
                site_ids.push_back(id);
            }
            for(const std::string_view& site : model::noncontact_site_list) {
                std::string site_str = std::string(site);
                int id = mj_name2id(mj_model, mjOBJ_SITE, site_str.data());
                assert(id != -1 && "Site not found in model.");
                noncontact_sites.push_back(site_str);
                noncontact_site_ids.push_back(id);
            }
            for(const std::string_view& site : model::contact_site_list) {
                std::string site_str = std::string(site);
                int id = mj_name2id(mj_model, mjOBJ_SITE, site_str.data());
                assert(id != -1 && "Site not found in model.");
                contact_sites.push_back(site_str);
                contact_site_ids.push_back(id);
            }
            for(const std::string_view& body : model::body_list) {
                std::string body_str = std::string(body);
                int id = mj_name2id(mj_model, mjOBJ_BODY, body_str.data());
                assert(id != -1 && "Body not found in model.");
                bodies.push_back(body_str);
                body_ids.push_back(id);
            }
            // Assert Number of Sites and Bodies are equal:
            assert(site_ids.size() == body_ids.size() && "Number of Sites and Bodies must be equal.");

            // Set initial state to initialize the optimization:
            state = initial_state;
            initialized = true;

            return absl::OkStatus();
        }

        absl::Status initialize_optimization() {
            if(!initialized)
                return absl::FailedPreconditionError("Operational Space Controller not initialized.");

            // Initialize mj_data with initial state:
            update_mj_data();

            // Initialize Optimization:
            absl::Status result = set_up_optimization();
            if(!result.ok())
                return result;

            optimization_initialized = true;

            return absl::OkStatus();
        }

        absl::Status initialize_thread() {
            if(!initialized || !optimization_initialized)
                return absl::FailedPreconditionError("Initialization precoditions not met. Initialize controller and optimization before starting control thread.");
            
            thread = std::thread(&OperationalSpaceController::control_loop, this);
            thread_initialized = true;
            return absl::OkStatus();
        }

        absl::Status stop_thread() {
            if(!thread_initialized)
                return absl::FailedPreconditionError("Operation Space Control Thread not initialized");

            running = false;
            thread.join();
            return absl::OkStatus();
        }

        bool is_initialized() {
            return initialized;
        }

        bool is_optimization_initialized() {
            return optimization_initialized;
        }

        bool is_thread_initialized() {
            return thread_initialized;
        }

        absl::Status clean_up() {
            if(!initialized)
                return absl::FailedPreconditionError("Operational Space Controller not initialized. Nothing to clean up");

            mj_deleteData(mj_data);
            mj_deleteModel(mj_model);

            return absl::OkStatus();
        }

        void update_state(const State& new_state) {
            std::lock_guard<std::mutex> lock(mutex);
            state = new_state;
        }

        void update_taskspace_targets(const TaskspaceTargets& new_taskspace_targets) {
            std::lock_guard<std::mutex> lock(mutex);
            taskspace_targets = new_taskspace_targets;
        }

        Vector<model::nu_size> get_torque_command() {
            std::lock_guard<std::mutex> lock(mutex);
            return torque_command;
        }

        Vector<optimization::design_vector_size> get_solution() {
            std::lock_guard<std::mutex> lock(mutex);
            return solution;
        }

        private:
            // Shared Variables: (Inputs: state and taskspace_targets) (Outputs: torque_command)
            State state;
            Matrix<model::site_ids_size, 6> taskspace_targets = Matrix<model::site_ids_size, 6>::Zero();
            Vector<model::nu_size> torque_command = Vector<model::nu_size>::Zero();
            /* Initialization Flags */
            bool initialized = false;
            bool optimization_initialized = false;
            bool thread_initialized = false;
            /* Mujoco Variables */
            mjModel* mj_model;
            mjData* mj_data;
            std::filesystem::path xml_path;
            std::vector<std::string> sites;
            std::vector<std::string> bodies;
            std::vector<std::string> noncontact_sites;
            std::vector<std::string> contact_sites;
            std::vector<int> site_ids;
            std::vector<int> noncontact_site_ids;
            std::vector<int> contact_site_ids;
            std::vector<int> body_ids;
            Matrix<model::site_ids_size, 3> points;
            // Matrix<model::site_ids_size, 3> points2;
            static constexpr bool is_fixed_based = false;
            // Control Thread:
            int control_rate_us;
            std::atomic<bool> running{true};
            std::mutex mutex;
            std::thread thread;
            /* OSQP Solver, settings, and matrices */
            OsqpInstance instance;
            OsqpSolver solver;
            OsqpSettings settings;
            OsqpExitCode exit_code;
            Vector<optimization::design_vector_size> solution = Vector<optimization::design_vector_size>::Zero();
            Vector<optimization::constraint_matrix_rows> dual_solution = Vector<optimization::constraint_matrix_rows>::Zero();
            Vector<optimization::design_vector_size> design_vector = Vector<optimization::design_vector_size>::Zero();
            const double infinity = OSQP_INFTY;
            OSCData osc_data;
            OptimizationData opt_data;
            const float big_number = 1e4;
            // Constraints:
            MatrixColMajor<optimization::design_vector_size, optimization::design_vector_size> Abox = 
                MatrixColMajor<optimization::design_vector_size, optimization::design_vector_size>::Identity();
            Vector<optimization::dv_size> dv_lb = Vector<optimization::dv_size>::Constant(-infinity);
            Vector<optimization::dv_size> dv_ub = Vector<optimization::dv_size>::Constant(infinity);
            // =========== walter sr - motors: =============================================================================
            // hip - link: https://mad-motor.com/products/mad-components-m6c12-eee-industrial-drone-motor?VariantsId=10508
            // Hip motor has no gear reduction; 'transfer gears' are 1:1.
            // MAX TORQUE 2.64 Nm
            // knee - link: https://mjbots.com/products/mj5208
            // Knee motor runs through a 19:30 pulley reduction.
            // PEAK TORQUE 1.7 Nm (with 19:30 gear reduction - 1.076666)
            // =========== motor names: ====================================================================================
            // actuators/control inputs - [bl_hip/knee,br,fl,fr] - [torso_left_thigh_joint/shin,right,head_left,right]
            // Vector<model::nu_size> u_lb = {
            //     -2.64, -1.076,
            //     -2.64, -1.076,
            //     -2.64, -1.076,
            //     -2.64, -1.076
            // };
            // Vector<model::nu_size> u_ub = {
            //     2.64, 1.076,
            //     2.64, 1.076,
            //     2.64, 1.076,
            //     2.64, 1.076
            // };
            Vector<model::nu_size> u_lb = {
                -1000, -1000,
                -1000, -1000,
                -1000, -1000,
                -1000, -1000
            };
            Vector<model::nu_size> u_ub = {
                1000, 1000,
                1000, 1000,
                1000, 1000,
                1000, 1000
            };
            // Vector<model::nu_size> u_lb = {
            //     -100, -100,
            //     -100, -100,
            //     -100, -100,
            //     -100, -100
            // };
            // Vector<model::nu_size> u_ub = {
            //     100, 100,
            //     100, 100,
            //     100, 100,
            //     100, 100
            // };            
            Vector<optimization::z_size> z_lb = {
                -infinity, -infinity, 0.0,
                -infinity, -infinity, 0.0,
                -infinity, -infinity, 0.0,
                -infinity, -infinity, 0.0,
                -infinity, -infinity, 0.0,
                -infinity, -infinity, 0.0,
                -infinity, -infinity, 0.0,
                -infinity, -infinity, 0.0
            };
            Vector<optimization::z_size> z_ub = {
                infinity, infinity, big_number,
                infinity, infinity, big_number,
                infinity, infinity, big_number,
                infinity, infinity, big_number,
                infinity, infinity, big_number,
                infinity, infinity, big_number,
                infinity, infinity, big_number,
                infinity, infinity, big_number
            };
            Vector<optimization::bineq_sz> bineq_lb = Vector<optimization::bineq_sz>::Constant(-infinity);
            
            absl::Status set_up_optimization() {
                // Initialize the Optimization: (Everything should be Column Major for OSQP)
                // Get initial data from initial state:
                update_osc_data();
                update_optimization_data();

                // Concatenate Constraint Matrix:
                MatrixColMajor<optimization::constraint_matrix_rows, optimization::constraint_matrix_cols> A;
                A << opt_data.Aeq, opt_data.Aineq, Abox;
                // Calculate Bounds:
                Vector<optimization::bounds_size> lb;
                Vector<optimization::bounds_size> ub;
                Vector<optimization::z_size> z_lb_masked = z_lb;
                Vector<optimization::z_size> z_ub_masked = z_ub;
                for(int i = 0; i < model::contact_site_ids_size; i++) {
                    z_lb_masked(Eigen::seqN(3 * i, 3)) *= state.contact_mask(i);
                    z_ub_masked(Eigen::seqN(3 * i, 3)) *= state.contact_mask(i);
                }
                lb << opt_data.beq, bineq_lb, dv_lb, u_lb, z_lb_masked;
                ub << opt_data.beq, opt_data.bineq, dv_ub, u_ub, z_ub_masked;
                
                // Initialize Sparse Matrix:
                Eigen::SparseMatrix<double> sparse_H = opt_data.H.sparseView();
                Eigen::SparseMatrix<double> sparse_A = A.sparseView();
                sparse_H.makeCompressed();
                sparse_A.makeCompressed();

                // Initalize OSQP workspace:
                instance.objective_matrix = sparse_H;
                instance.objective_vector = opt_data.f;
                instance.constraint_matrix = sparse_A;
                instance.lower_bounds = lb;
                instance.upper_bounds = ub;
                
                // Check initialization:
                absl::Status result = solver.Init(instance, settings);
                return result;
            }

            void update_mj_data() {
                Vector<model::nq_size> qpos = Vector<model::nq_size>::Zero();
                Vector<model::nv_size> qvel = Vector<model::nv_size>::Zero();
                if constexpr (is_fixed_based) {
                    qpos = state.motor_position;
                    qvel = state.motor_velocity;
                } 
                else {
                    const Vector<3> zero_vector = {0.0, 0.0, 0.0};
                    qpos << zero_vector, state.body_rotation, state.motor_position;
                    qvel << state.linear_body_velocity, state.angular_body_velocity, state.motor_velocity;
                }

                // Update Mujoco Data:
                mj_data->qpos = qpos.data();
                mj_data->qvel = qvel.data();

                // Runs steps: 2-12, 12-18:
                mj_fwdPosition(mj_model, mj_data);
                mj_fwdVelocity(mj_model, mj_data);
                 

                // Update Points:
                points= Eigen::Map<Matrix<model::site_ids_size, 3>>(mj_data->site_xpos)(site_ids, Eigen::placeholders::all);
                
                // points2= Eigen::Map<Matrix<model::site_ids_size, 3>>(mj_data->site_xpos);
                // points(site_ids, Eigen::placeholder::all);
                // Eigen::Matrix<double, model::site_ids_size, 3> points;
                // for (int i = 0; i < model::site_ids_size; ++i) {
                //     int site_index = site_ids[i];
                //     points.row(i) = Eigen::Vector3d(mj_data->site_xpos[3 * site_index + 0],
                //                                     mj_data->site_xpos[3 * site_index + 1],
                //                                     mj_data->site_xpos[3 * site_index + 2]);
                //   }             
                // points = Eigen::Map<Matrix<model::site_ids_size, 3>>(mj_data->site_xpos)(site_ids, Eigen::placeholder::all);

                // std::cout << "points2 - unordered: " << points2 << std::endl;
                // std::cout << "points - REORDERED: " << points << std::endl;
            }

            void update_osc_data() {
                // Mass Matrix:
                Matrix<model::nv_size, model::nv_size> mass_matrix = 
                    Matrix<model::nv_size, model::nv_size>::Zero();
                mj_fullM(mj_model, mass_matrix.data(), mj_data->qM);
    
                // Coriolis Matrix:
                Vector<model::nv_size> coriolis_matrix = 
                    Eigen::Map<Vector<model::nv_size>>(mj_data->qfrc_bias);
    
                // Generalized Positions and Velocities:
                Vector<model::nq_size> generalized_positions = 
                    Eigen::Map<Vector<model::nq_size> >(mj_data->qpos);
                Vector<model::nv_size> generalized_velocities = 
                    Eigen::Map<Vector<model::nv_size>>(mj_data->qvel);
    
                // Jacobian Calculation:
                Matrix<optimization::p_size, model::nv_size> jacobian_translation = 
                    Matrix<optimization::p_size, model::nv_size>::Zero();
                Matrix<optimization::r_size, model::nv_size> jacobian_rotation = 
                    Matrix<optimization::r_size, model::nv_size>::Zero();
                Matrix<optimization::p_size, model::nv_size> jacobian_dot_translation = 
                    Matrix<optimization::p_size, model::nv_size>::Zero();
                Matrix<optimization::r_size, model::nv_size> jacobian_dot_rotation = 
                    Matrix<optimization::r_size, model::nv_size>::Zero();
                for (int i = 0; i < model::body_ids_size; i++) {
                    // Temporary Jacobian Matrices:
                    Matrix<3, model::nv_size> jacp = Matrix<3, model::nv_size>::Zero();
                    Matrix<3, model::nv_size> jacr = Matrix<3, model::nv_size>::Zero();
                    Matrix<3, model::nv_size> jacp_dot = Matrix<3, model::nv_size>::Zero();
                    Matrix<3, model::nv_size> jacr_dot = Matrix<3, model::nv_size>::Zero();
    
                    // Calculate Jacobian:
                    mj_jac(mj_model, mj_data, jacp.data(), jacr.data(), points.row(i).data(), body_ids[i]);
    
                    // Calculate Jacobian Dot:
                    mj_jacDot(mj_model, mj_data, jacp_dot.data(), jacr_dot.data(), points.row(i).data(), body_ids[i]);
    
                    // Append to Jacobian Matrices:
                    int row_offset = i * 3;
                    for(int row_idx = 0; row_idx < 3; row_idx++) {
                        for(int col_idx = 0; col_idx < model::nv_size; col_idx++) {
                            jacobian_translation(row_idx + row_offset, col_idx) = jacp(row_idx, col_idx);
                            jacobian_rotation(row_idx + row_offset, col_idx) = jacr(row_idx, col_idx);
                            jacobian_dot_translation(row_idx + row_offset, col_idx) = jacp_dot(row_idx, col_idx);
                            jacobian_dot_rotation(row_idx + row_offset, col_idx) = jacr_dot(row_idx, col_idx);
                        }
                    }
                }
    
                // Stack Jacobian Matrices: Taskspace Jacobian: [jacp; jacr], Jacobian Dot: [jacp_dot; jacr_dot]
                Matrix<optimization::s_size, model::nv_size> taskspace_jacobian = Matrix<optimization::s_size, model::nv_size>::Zero();
                Matrix<optimization::s_size, model::nv_size> jacobian_dot = Matrix<optimization::s_size, model::nv_size>::Zero();
                taskspace_jacobian << jacobian_translation, jacobian_rotation;
                jacobian_dot << jacobian_dot_translation, jacobian_dot_rotation;
    
                // Calculate Taskspace Bias Acceleration:
                Vector<optimization::s_size> taskspace_bias = Vector<optimization::s_size>::Zero();
                taskspace_bias = jacobian_dot * generalized_velocities;
    
                // Contact Jacobian: Shape (NV, 3 * num_contacts) 
                // This assumes contact frames are the last rows of the translation component of the taskspace_jacobian (jacobian_translation).
                // contact_jacobian = jacobian_translation[end-(3 * contact_site_ids_size):end, :].T
                Matrix<model::nv_size, optimization::z_size> contact_jacobian = 
                    Matrix<model::nv_size, optimization::z_size>::Zero();
    
                contact_jacobian = jacobian_translation(
                    Eigen::seq(Eigen::placeholders::end - Eigen::fix<optimization::z_size>, Eigen::placeholders::last),
                    Eigen::placeholders::all
                ).transpose();
    
                // Assign to OSCData:
                osc_data.mass_matrix = mass_matrix;
                osc_data.coriolis_matrix = coriolis_matrix;
                osc_data.contact_jacobian = contact_jacobian;
                osc_data.taskspace_jacobian = taskspace_jacobian;
                osc_data.taskspace_bias = taskspace_bias;
                osc_data.previous_q = generalized_positions;
                osc_data.previous_qd = generalized_velocities;
            }
    
            void update_optimization_data() {
                // Convert OSCData to Column Major for Casadi Functions:
                auto mass_matrix = matrix_utils::transformMatrix<double, model::nv_size, model::nv_size, matrix_utils::ColumnMajor>(osc_data.mass_matrix.data());
                auto coriolis_matrix = matrix_utils::transformMatrix<double, model::nv_size, 1, matrix_utils::ColumnMajor>(osc_data.coriolis_matrix.data());
                auto contact_jacobian = matrix_utils::transformMatrix<double, model::nv_size, optimization::z_size, matrix_utils::ColumnMajor>(osc_data.contact_jacobian.data());
                auto taskspace_jacobian = matrix_utils::transformMatrix<double, optimization::s_size, model::nv_size, matrix_utils::ColumnMajor>(osc_data.taskspace_jacobian.data());
                auto taskspace_bias = matrix_utils::transformMatrix<double, optimization::s_size, 1, matrix_utils::ColumnMajor>(osc_data.taskspace_bias.data());
                auto desired_taskspace_ddx = matrix_utils::transformMatrix<double, model::site_ids_size, 6, matrix_utils::ColumnMajor>(taskspace_targets.data());
                
                // Evaluate Casadi Functions:
                auto Aeq_matrix = evaluate_function<AeqParams>(Aeq_ops, {design_vector.data(), mass_matrix.data(), coriolis_matrix.data(), contact_jacobian.data()});
                auto beq_matrix = evaluate_function<beqParams>(beq_ops, {design_vector.data(), mass_matrix.data(), coriolis_matrix.data(), contact_jacobian.data()});
                auto Aineq_matrix = evaluate_function<AineqParams>(Aineq_ops, {design_vector.data()});
                auto bineq_matrix = evaluate_function<bineqParams>(bineq_ops, {design_vector.data()});
                auto H_matrix = evaluate_function<HParams>(H_ops, {design_vector.data(), desired_taskspace_ddx.data(), taskspace_jacobian.data(), taskspace_bias.data()});
                auto f_matrix = evaluate_function<fParams>(f_ops, {design_vector.data(), desired_taskspace_ddx.data(), taskspace_jacobian.data(), taskspace_bias.data()});
    
                // Assign to OptimizationData:
                opt_data.H = H_matrix;
                opt_data.f = f_matrix;
                opt_data.Aeq = Aeq_matrix;
                opt_data.beq = beq_matrix;
                opt_data.Aineq = Aineq_matrix;
                opt_data.bineq = bineq_matrix;
            }
            
            absl::Status update_optimization() {
                // Concatenate Constraint Matrix:
                MatrixColMajor<optimization::constraint_matrix_rows, optimization::constraint_matrix_cols> A;
                A << opt_data.Aeq, opt_data.Aineq, Abox;
                // Calculate Bounds:
                Vector<optimization::bounds_size> lb;
                Vector<optimization::bounds_size> ub;
                Vector<optimization::z_size> z_lb_masked = z_lb;
                Vector<optimization::z_size> z_ub_masked = z_ub;
                for(int i = 0; i < model::contact_site_ids_size; i++) {
                    z_lb_masked(Eigen::seqN(3 * i, 3)) *= state.contact_mask(i);
                    z_ub_masked(Eigen::seqN(3 * i, 3)) *= state.contact_mask(i);
                }
                lb << opt_data.beq, bineq_lb, dv_lb, u_lb, z_lb_masked;
                ub << opt_data.beq, opt_data.bineq, dv_ub, u_ub, z_ub_masked;
                
                // Initialize Sparse Matrix:
                Eigen::SparseMatrix<double> sparse_H = opt_data.H.sparseView();
                Eigen::SparseMatrix<double> sparse_A = A.sparseView();
                sparse_H.makeCompressed();
                sparse_A.makeCompressed();

                // Check if sparisty changed:
                absl::Status result;
                auto sparsity_check = solver.UpdateObjectiveAndConstraintMatrices(sparse_H, sparse_A);
                if(sparsity_check.ok()) {
                    // Update Internal OSQP workspace:
                    result.Update(solver.SetObjectiveVector(opt_data.f));
                    result.Update(solver.SetBounds(lb, ub));
                }
                else {
                    // Reinitalize OSQP workspace:
                    instance.objective_matrix = sparse_H;
                    instance.objective_vector = opt_data.f;
                    instance.constraint_matrix = sparse_A;
                    instance.lower_bounds = lb;
                    instance.upper_bounds = ub;
                    
                    // Return type is absl::Status
                    result.Update(solver.Init(instance, settings));
                    
                    // Setwarmstart:
                    result.Update(solver.SetWarmStart(solution, dual_solution));
                }

                return result;
            }
    
            void solve_optimization() {
                // Solve the Optimization:
                exit_code = solver.Solve();
                solution = solver.primal_solution();
                dual_solution = solver.dual_solution();
            }
    
            void reset_optimization() {
                // Set Warm Start to Zero:
                Vector<optimization::constraint_matrix_cols> primal_vector = Vector<optimization::constraint_matrix_cols>::Zero();
                Vector<optimization::constraint_matrix_rows> dual_vector = Vector<optimization::constraint_matrix_rows>::Zero();
                std::ignore = solver.SetWarmStart(primal_vector, dual_vector);
            }

            /* Consistent Execution Time: */
            void control_loop() {
                using Clock = std::chrono::steady_clock;
                auto next_time = Clock::now();
                // Thread Loop:
                while(running) {
                    // Calculate next execution time first
                    next_time += std::chrono::microseconds(control_rate_us);

                    /* Lock Guard Scope */
                    {   
                        std::lock_guard<std::mutex> lock(mutex);
                        // Update Mujoco Data:
                        update_mj_data();

                        // Get OSC Data:
                        update_osc_data();

                        // Get Optimization Data:
                        update_optimization_data();

                        // Update Optimization: (No error handling for now)
                        std::ignore = update_optimization();

                        // Solve Optimization:
                        solve_optimization();
                        
                        // Get torques from QP solution:
                        torque_command = solution(Eigen::seqN(optimization::dv_idx, optimization::u_size));
                    }
                    // Check for overrun and sleep until next execution time
                    auto now = Clock::now();
                    if (now < next_time) {
                        std::this_thread::sleep_until(next_time);
                    } 
                    else {
                        // Log overrun
                        auto overrun = std::chrono::duration_cast<std::chrono::microseconds>(now - next_time);
                        std::cout << "Operational Space Control Loop Execution Time Exceeded Control Rate: " 
                                << overrun.count() << "us" << std::endl;
                        // Reset next execution time to prevent cascading delays
                        next_time = now;
                    }
                }
            }
};
