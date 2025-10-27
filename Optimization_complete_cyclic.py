from abaqus import *
import job
from fontTools.misc.bezierTools import epsilon
from odbAccess import *
from abaqusConstants import *
from odbMaterial import *
from odbSection import *

import sketch
import part
import assembly
import step
import load
import mesh
import optimization
import job
import visualization
import connectorBehavior
import regionToolset

import sys
import numpy as np
import os
import glob
import json
import csv

from copy import deepcopy

# path to location of EasyPBC plug-in
sys.path.insert(0,'/home/rzlin/ib92ifar/abaqus_plugins/EasyPBC V.1.4')
import easypbc

from scipy.optimize import minimize

#read input file and create CubeParameter object
def read_parameter_file(filename, input_directory):
    with open(filename, 'r') as file:
        input_data = json.load(file)

    params = CubeParameters(
        model_name=input_data['Modelname'],
        part_name=f"{input_data['Modelname']}_Part",
        instance_name=f"{input_data['Modelname']}_Instance",
        section_name=f"{input_data['Modelname']}_Section",
        cube_size=input_data['CubeSize'],
        material_name=input_data['MaterialName'],
        material_type=input_data['MaterialType'],
        youngs_modulus_bounds=np.array([input_data['YoungsModulus']['min'], input_data['YoungsModulus']['max']]),
        youngs_modulus = input_data['YoungsModulus']['value'],  
        poisson_ratio_bounds=np.array([input_data['PoissonRatio']['min'], input_data['PoissonRatio']['max']]),
        poisson_ratio=input_data['PoissonRatio']['value'],  # Initial value
        plastic_yield_bounds=np.array([input_data['PlasticYield']['min'], input_data['PlasticYield']['max']]),
        plastic_yield=input_data['PlasticYield']['value'],  # Initial value
        alpha_bounds=np.array([input_data['Alpha']['min'], input_data['Alpha']['max']]),
        alpha=input_data['Alpha']['value'],  # Initial value
        beta_bounds=np.array([input_data['Beta']['min'], input_data['Beta']['max']]),
        beta=input_data['Beta']['value'],  # Initial value
        gamma_bounds=np.array([input_data['Gamma']['min'], input_data['Gamma']['max']]),
        gamma=input_data['Gamma']['value'],  # Initial value
        C10_bounds=np.array([input_data['C10']['min'], input_data['C10']['max']]),
        D1_bounds=np.array([input_data['D1']['min'], input_data['D1']['max']]),
        plastic_param_file=input_data['PlasticMaterialParameters'],
        element_number=input_data['NumberOfElementsPerEdge'],
        md_data_dict = {file_info["filename"]: file_info["weight"] for file_info in input_data["MDDataFiles"]},
        prescribed_directions= input_data['PrescribedDirections'],
        stress_analysis_direction=input_data['StressAnalysisDirection'],
        strain_analysis_direction=input_data['StrainAnalysisDirection'],
        weights=input_data['weights'],
        iteration_number=input_data['NumberOfIterations'],
        test_name=input_data['Testname'],
        input_dirctory = input_directory
    )

    return params

#create CubeParameters object to store input parameters
class CubeParameters:
    def __init__(self, model_name, part_name, instance_name, section_name,
                 cube_size, material_name, material_type,
                 youngs_modulus_bounds, youngs_modulus, poisson_ratio_bounds, poisson_ratio,
                 plastic_param_file, plastic_yield_bounds, plastic_yield,
                 alpha_bounds, alpha, beta_bounds, beta,
                 gamma_bounds, gamma, C10_bounds, D1_bounds,
                 element_number, md_data_dict, prescribed_directions,
                 stress_analysis_direction, strain_analysis_direction, weights, iteration_number, test_name, input_dirctory):
        
        self.input_dirctory = input_dirctory
        self.model_name = model_name
        self.part_name = part_name
        self.instance_name = instance_name
        self.section_name = section_name
        self.cube_size = cube_size
        self.material_name = material_name
        self.material_type = material_type
        self.youngs_modulus_bounds = youngs_modulus_bounds  
        self.youngs_modulus = youngs_modulus
        self.poisson_ratio_bounds = poisson_ratio_bounds    
        self.poisson_ratio = poisson_ratio
        self.plastic_param_file = plastic_param_file
        self.plastic_strain = self.read_plastic_strain()
        self.number_plast_values = len(self.plastic_strain)
        self.plastic_yield_bounds = plastic_yield_bounds 
        self.plastic_yield = np.array(plastic_yield) 
        self.alpha_bounds = alpha_bounds  
        self.alpha = np.array(alpha)  
        self.beta_bounds = beta_bounds    
        self.beta = np.array(beta)  
        self.gamma_bounds = gamma_bounds  
        self.gamma = np.array(gamma) 
        self.C10_bounds = C10_bounds
        self.D1_bounds = D1_bounds

        self.element_number = element_number
        self.md_data_dict = md_data_dict
        self.prescribed_directions = prescribed_directions
        self.stress_analysis_direction = stress_analysis_direction
        self.strain_analysis_direction = strain_analysis_direction
        self.weights = weights
        self.iteration_number = iteration_number
        self.test_name = test_name

        self.shear_modulus = self.youngs_modulus/ (2*(1 + self.poisson_ratio))
        self.C10 = self.shear_modulus / 2
        self.bulk_modulus = self.youngs_modulus / (3 * (1- 2 * self.poisson_ratio))
        self.D1 = 2 / self.bulk_modulus 

        self.C10_scaled = ((self.C10 - self.C10_bounds[0]) /
        (self.C10_bounds[1] -self.C10_bounds[0]))

        self.D1_scaled = ((self.D1 - self.D1_bounds[0]) /
        (self.D1_bounds[1] -self.D1_bounds[0]))

        self.youngs_modulus_scaled = ((self.youngs_modulus - self.youngs_modulus_bounds[0]) /
        (self.youngs_modulus_bounds[1] -self.youngs_modulus_bounds[0]))

        self.poisson_ratio_scaled = ((self.poisson_ratio - self.poisson_ratio_bounds[0] ) /
        (self.poisson_ratio_bounds[1] - self.poisson_ratio_bounds[0]))

        self.plastic_yield_scaled = ((self.plastic_yield - self.plastic_yield_bounds[0]) /
        (self.plastic_yield_bounds[1] - self.plastic_yield_bounds[0]))

        self.alpha_scaled = ((self.alpha - self.alpha_bounds[0]) / 
        (self.alpha_bounds[1] -self.alpha_bounds[0]))

        self.beta_scaled = ((self.beta - self.beta_bounds[0]) /
        (self.beta_bounds[1] -self.beta_bounds[0]))

        self.gamma_scaled = ((self.gamma - self.gamma_bounds[0]) /
        (self.gamma_bounds[1] -self.gamma_bounds[0]))

        self.number_of_params = self.check_array_lengths()

    #read plastic strain values to apply them as amplitude
    def read_plastic_strain(self):
        input_file_path = os.path.join(self.input_dirctory, self.plastic_param_file)
        data = np.loadtxt(input_file_path, delimiter=' ', skiprows=1)  
        strain = data[:, 1]  
        return strain
    
    # checl if all material parameter arrays have the same length
    def check_array_lengths(self):
        arrays = [
            self.plastic_yield,
            self.alpha,
            self.beta,
            self.gamma
        ]

        lengths = [len(array) for array in arrays]  
        if all(length == lengths[0] for length in lengths):
            print(f"All arrays have the same length: {lengths[0]}")
            return lengths[0]
        else:
            raise ValueError("Arrays have the different length.")

#create reference data object
class MDData:
    def __init__(self, filename, input_directory):
        self.filename = filename
        self.input_directory = input_directory
        self.data = self.read_md_file()
       
    # function to read md-data from file
    def read_md_file(self):  
        md_path = os.path.join(self.input_directory, self.filename)
        with open(md_path, 'r') as file:
            next(file) 
            md_data = {
                "step": [],
                "strain_xx": [],
                "strain_yy": [],
                "strain_zz": [],
                "strain_xy": [],
                "strain_xz": [],
                "strain_yz": [],
                "stress_xx": [],
                "stress_yy": [],
                "stress_zz": [],
                "stress_xy": [],
                "stress_xz": [],
                "stress_yz": []
            }

            # column readout depends on md-data file layout
            for line in file:
                values = line.split()
                md_data["step"].append(float(values[0]))
                md_data["strain_xx"].append(float(values[1]))
                md_data["strain_yy"].append(float(values[2]))
                md_data["strain_zz"].append(float(values[3]))
                md_data["strain_xy"].append(float(values[4]))
                md_data["strain_xz"].append(float(values[5]) if len(values) > 5 else 0.0) 
                md_data["strain_yz"].append(float(values[6]) if len(values) > 6 else 0.0)
                md_data["stress_xx"].append(float(values[7]))
                md_data["stress_yy"].append(float(values[8]) if len(values) > 8 else 0.0)
                md_data["stress_zz"].append(float(values[9]) if len(values) > 9 else 0.0)
                md_data["stress_xy"].append(float(values[10]) if len(values) > 10 else 0.0)
                md_data["stress_xz"].append(float(values[11]) if len(values) > 11 else 0.0)
                md_data["stress_yz"].append(float(values[12]) if len(values) > 12 else 0.0)

        return md_data
    
#create Cube object in Abaqus
class MDBCube:
    def __init__(self, cube_parameters, filename, work_directory, direction):
        self.parameters = deepcopy(cube_parameters)
        self.filename = filename
        self.work_directory = work_directory
        self.md_data = self.read_md_file(filename)  
        self.parameters.model_name = self.rename_model(self.parameters.model_name, self.filename, direction)
        
    def read_md_file(self, filename):
        with open(filename, 'r') as file:
            next(file)
            md_data = {
                "step": [],
                "strain_xx": [],
                "strain_yy": [],
                "strain_zz": [],
                "strain_xy": [],
                "strain_xz": [],
                "strain_yz": [],
                "stress_xx": [],
                "stress_yy": [],
                "stress_zz": [],
                "stress_xy": [],
                "stress_xz": [],
                "stress_yz": []
            }

            # column readout depends on md-data file layout
            for line in file:
                values = line.split()
                md_data["step"].append(float(values[0]))
                md_data["strain_xx"].append(float(values[1]))
                md_data["strain_yy"].append(float(values[2]))
                md_data["strain_zz"].append(float(values[3]))
                md_data["strain_xy"].append(float(values[4]))
                md_data["strain_xz"].append(float(values[5]) if len(values) > 5 else 0.0)  # Addresses possible missing data
                md_data["strain_yz"].append(float(values[6]) if len(values) > 6 else 0.0)
                md_data["stress_xx"].append(float(values[7]))
                md_data["stress_yy"].append(float(values[8]) if len(values) > 8 else 0.0)
                md_data["stress_zz"].append(float(values[9]) if len(values) > 9 else 0.0)
                md_data["stress_xy"].append(float(values[10]) if len(values) > 10 else 0.0)
                md_data["stress_xz"].append(float(values[11]) if len(values) > 11 else 0.0)
                md_data["stress_yz"].append(float(values[12]) if len(values) > 12 else 0.0)

        return md_data

    # rename the model corresponding to the testname in the inputfile
    def rename_model(self,base_model_name, filename, direction):
        #base_model_name = parameters.model_name
        modefied_file_name = filename.replace('.', '_')
        short_file_name = modefied_file_name[-9:-4]
        modified_model_name = f"{base_model_name}_{short_file_name}_{direction} "
        return modified_model_name
    
    # create model
    def create_cube(self, i):
        # create skecth, part and model
        my_model = mdb.Model(name=self.parameters.model_name)
        sketch = my_model.ConstrainedSketch(name='__profile__', sheetSize=2 * self.parameters.cube_size)
        sketch.rectangle(point1=(0, 0), point2=(self.parameters.cube_size, self.parameters.cube_size))
        my_part = my_model.Part(name=self.parameters.model_name, dimensionality=THREE_D, type=DEFORMABLE_BODY)
        my_part.BaseSolidExtrude(sketch=sketch, depth=self.parameters.cube_size)
        my_part.Set(faces=my_part.faces, name='CubeFaces')

        # create material and assign it to a section
        my_material = mdb.models[self.parameters.model_name].Material(name=self.parameters.material_name)
        my_model.materials[self.parameters.material_name].Elastic(
            table=((self.parameters.youngs_modulus, self.parameters.poisson_ratio),))
        self.create_plastic_material(my_material, i)
        my_model.HomogeneousSolidSection(material=self.parameters.material_name, name=self.parameters.section_name,
                                         thickness=None)
        my_region = my_part.Set(cells=my_part.cells[:], name='Entire_Part')
        my_part.SectionAssignment(region=my_region, sectionName=self.parameters.section_name, offset=0.0,
                                  offsetField='', thicknessAssignment=FROM_SECTION)
        print(f"Section '{self.parameters.section_name}' assigned to part '{self.parameters.part_name}'.")

        # create instance and mesh part
        my_instance = my_model.rootAssembly.Instance(name=self.parameters.instance_name, part=my_part, dependent=OFF)
        element_size = self.parameters.cube_size / self.parameters.element_number
        my_model.rootAssembly.seedPartInstance(regions=(my_instance,), size=element_size, deviationFactor=element_size,
                                               minSizeFactor=0.1)
        my_model.rootAssembly.generateMesh(regions=(my_instance,))

        # create node- and elementsets at the surfaces
        for direction in ['xx', 'yy', 'zz']:
            coord_direction = ['xx', 'yy', 'zz'].index(direction)
            self.node_set_max_coord( coord_direction, f'Max_{direction}_Nodes', 1)
            self.node_set_max_coord( coord_direction, f'Min_{direction}_Nodes', -1)
            self.element_set_max_coord( coord_direction, f'Max_{direction}_Elements', 1)
            self.element_set_max_coord( coord_direction, f'Min_{direction}_Elements', -1)
        print(f"Cube model '{self.parameters.model_name}' created and Element- and Nodesets generated.")
        return my_model

    # search coordinate of surface node
    def search_maximum_node_coordinate(self, coord_direction, coord_id):
        target_coord = float('-inf') if coord_id == 1 else float('inf')
        instance = mdb.models[self.parameters.model_name].rootAssembly.instances[self.parameters.instance_name]
        for node in instance.nodes: 
            act_coord = node.coordinates[coord_direction]
            if coord_id == 1 and act_coord > target_coord:
                target_coord = act_coord
            elif coord_id == -1 and act_coord < target_coord:
                target_coord = act_coord
        return target_coord

    # create surface nodeset
    def node_set_max_coord(self,  coord_direction, node_set_name, coord_id):
        target_coord = self.search_maximum_node_coordinate( coord_direction, coord_id)
        matching_nodes = []
        instance = mdb.models[self.parameters.model_name].rootAssembly.instances[self.parameters.instance_name]
        for node in instance.nodes: 
            if node.coordinates[coord_direction] == target_coord:
                matching_nodes.append(node.label)
        if matching_nodes:
            created_set = mdb.models[self.parameters.model_name].rootAssembly.SetFromNodeLabels(
                name=node_set_name,
                nodeLabels=((self.parameters.instance_name, matching_nodes),)
            )
            print(f"Nodeset '{node_set_name}' with {len(matching_nodes)} nodes created.")
            return created_set
        else:
            print("Did not found matching node.")
            return None

    # create surface elementset
    def element_set_max_coord(self,  coord_direction, element_set_name, coord_id):
        target_coord = self.search_maximum_node_coordinate( coord_direction, coord_id)
        matching_elements = set()
        instance = mdb.models[self.parameters.model_name].rootAssembly.instances[self.parameters.instance_name]
        for element in instance.elements: 
            for node_index in element.connectivity:
                node = instance.nodes[node_index]
                if node.coordinates[coord_direction] == target_coord:
                    matching_elements.add(element.label)
                    break
        if matching_elements:
            created_set = mdb.models[self.parameters.model_name].rootAssembly.SetFromElementLabels(
                name=element_set_name,
                elementLabels=((self.parameters.instance_name, list(matching_elements)),)
            )
            print(f"Elementset '{element_set_name}' with {len(matching_elements)} elements generated.")
            return created_set
        else:
            print("Did not found matching element.")
            return None
    
    # create plastic material section
    def create_plastic_material(self, material, n):
        plastic_stresses = [] 
        plastic_stresses = self.parameters.plastic_yield[n] + self.parameters.alpha[n] * (1 - np.exp(-self.parameters.beta[n] * self.parameters.plastic_strain)) + self.parameters.gamma[n] * self.parameters.plastic_strain
        plastic_data = []
        for m in range(self.parameters.number_plast_values):
            plastic_data.append(( plastic_stresses[m], self.parameters.plastic_strain[m],))
        material.Plastic(table=plastic_data)
        return None

    # create amplitude for load application
    def create_amplitude(self, direction):
        strain_map = {
        'E11': ('strain_xx', 'StrainXXAmplitude'),
        'E22': ('strain_yy', 'StrainYYAmplitude'),
        'E33': ('strain_zz', 'StrainZZAmplitude'),
        'G12': ('strain_xy', 'StrainXYAmplitude'),
        'G13': ('strain_xz', 'StrainXZAmplitude'),
        'G23': ('strain_yz', 'StrainYZAmplitude')
        }

        if direction not in strain_map:
            raise ValueError(f"Invalid direction specified: {direction}")
        
        strain_key, amplitude_name = strain_map[direction]
        strain_data = self.md_data[strain_key]
        step_data = self.md_data["step"]
        amplitude_points = tuple(zip(step_data, strain_data))
        mdb.models[self.parameters.model_name].TabularAmplitude(
            name=amplitude_name,
            timeSpan=STEP,
            smooth=SOLVER_DEFAULT,
            data=amplitude_points
        )
        print(f"Amplitude '{amplitude_name}' with {len(amplitude_points)} points created.")
        return amplitude_name
    
    # update increment setting and FieldOutput
    def update_increment_size(self):
        step_times = self.md_data["step"]
        if len(step_times) < 2:
            print("Not enough step-values to calculate increment size.")
            return

        total_time = step_times[-1] 
        step_name = 'Step-1'
        mdb.models[self.parameters.model_name].steps[step_name].setValues(
            nlgeom=ON,
            initialInc=total_time*1e-2,
            maxNumInc=10000,
            maxInc = 1.0,
            noStop=OFF,
            timeIncrementationMethod=AUTOMATIC,
            timePeriod=total_time
        )
        time_points_name = 'OutputPoints'
        mdb.models[self.parameters.model_name].TimePoint(name=time_points_name, points=((step_times[0], step_times[-1] , 
        1.0), ))
        
        field_output_request = mdb.models[self.parameters.model_name].fieldOutputRequests['F-Output-1']
        field_output_request.setValues(variables=('S', 'PE', 'PEEQ', 'PEMAG', 'NE', 'LE', 'U', 'RF'))
        field_output_request.setValues(timePoint=time_points_name)
        return None

    # update boundary condition
    def update_boundary_condition(self, direction): 
        bc_map_u = {
        'E11': ('E11-1', 'u1'),
        'E22': ('E22-1', 'u2'),
        'E33': ('E33-1', 'u3'),
        'G12': ('G12-1', 'u1', 'u2'),
        'G13': ('G13-1', 'u1', 'u3'),
        'G23': ('G23-1', 'u2', 'u3')
        }
    
        if direction not in bc_map_u:
            raise ValueError(f"Invalid direction specified: {direction}")

        bc_name = bc_map_u[direction][0]
        displacements = bc_map_u[direction][1:]

        model = mdb.models[self.parameters.model_name]
        if bc_name in model.boundaryConditions:
            amplitude_name = self.create_amplitude(direction)

            set_values = {displacements[0]: 1.0} 
            #  for shear strain apply whole displacement in one direction
            if len(displacements) > 1:
                set_values[displacements[1]] = 0.0

            set_values['amplitude'] = amplitude_name
            model.boundaryConditions[bc_name].setValues(**set_values)
            print(f"Boundary Condition '{bc_name}' with amplitude '{amplitude_name}' updated in directions {displacements}.")
        else:
            print(f"Boundary Condition named '{bc_name}' not found.")

    # call easyPBC to create job
    def create_job(self, filename, direction):
        directions = {"E11": False, "E22": False, "E33": False, 
                  "G12": False, "G13": False, "G23": False}
        
        if direction in directions:
            directions[direction] = True
        else:
            raise ValueError(f"Invalid direction specified: {direction}")

        easypbc.feasypbc(part=self.parameters.model_name, inst=self.parameters.instance_name, meshsens=1E-07, CPU=1,
                          E11=directions["E11"], E22=directions["E22"], E33=directions["E33"], G12=directions["G12"], 
                          G13=directions["G13"], G23=directions["G23"],
                          onlyPBC=False, CTE=False, intemp=0, fntemp=100)
        modified_job_name = f"job-{direction}_{filename.replace('.dat', '')}"
        mdb.jobs.changeKey(fromName=f'job-{direction}', toName=modified_job_name)

        return modified_job_name

    # save MDB 
    def save_mdb(self):
        mdb_file_name = os.path.join(self.work_directory, f"{self.parameters.model_name}.mdb")
        mdb.saveAs(mdb_file_name) 
        if 'Model-1' in mdb.models:
            del mdb.models['Model-1']
        return None


# Optimisation function
def optimization_multiple_stresses(scaled_material, parameters, md_data, result_dir, evaluation_count):
    odb = None
    # variable to store rmse values for all jobs
    total_rmse = 0
    rmse_list = []
    # rescale material parameters
    material = []
    
    # for optimisation of elastic and plastic parameters use commented out lines
    material.append(parameters.youngs_modulus)
    material.append(parameters.poisson_ratio)
    #material.append(scaled_material[0] * (parameters.youngs_modulus_bounds[1] - parameters.youngs_modulus_bounds[0]) + parameters.youngs_modulus_bounds[0]) 
    #material.append(scaled_material[1] * (parameters.poisson_ratio_bounds[1] - parameters.poisson_ratio_bounds[0]) + parameters.poisson_ratio_bounds[0])   
    material.append(scaled_material[0] * (parameters.plastic_yield_bounds[1] - parameters.plastic_yield_bounds[0]) + parameters.plastic_yield_bounds[0]) 
    material.append(scaled_material[1] * (parameters.alpha_bounds[1] - parameters.alpha_bounds[0]) + parameters.alpha_bounds[0])    
    material.append(scaled_material[2] * (parameters.beta_bounds[1] - parameters.beta_bounds[0]) + parameters.beta_bounds[0])    
    material.append(scaled_material[3] * (parameters.gamma_bounds[1] - parameters.gamma_bounds[0]) + parameters.gamma_bounds[0])   
    material.append(parameters.C10) 
    material.append(parameters.D1)  

    print('Scaled material parameters in current evaluation:', scaled_material)
    print('Rescaled Material parameters in current evaluation:', material)
    
    # compute plastic stress values
    plastic_stresses = []
    plastic_stresses = material[2] + material[3] * (1 - np.exp(-material[4] * parameters.plastic_strain)) + material[5] * parameters.plastic_strain
    plastic_data = []
    for i in range(parameters.number_plast_values):
        plastic_data.append(( plastic_stresses[i], parameters.plastic_strain[i],))

    # create hyperelastic material
    for model_name in mdb.models.keys():
        model = mdb.models[model_name]
        del model.materials[parameters.material_name].elastic

        model.materials[parameters.material_name].Hyperelastic(
             materialType=ISOTROPIC, table=((material[6], material[7]), ), testData=OFF, type=
             NEO_HOOKE, volumetricResponse=VOLUMETRIC_DATA)
        
        model.materials[parameters.material_name].Plastic(
            scaleStress=None,
            table=plastic_data
        )
        print(
            f'Material parameters written for model "{model_name}": E = {material[0]}, '
            f'nu = {material[1]}, yield stress = {material[2]}, alpha = {material[3]}, beta = {material[4]}, gamma = {material[5]}, C10 = {material[6]}, D1 = {material[7]}')

    # check for lock-files
    lck_files = glob.glob('*.lck')
    if lck_files:
        for file in lck_files:
            try:
                os.remove(file)
            except Exception as e:
                print(f'Error when deleting {file}: {e}')
    else:
        print('No .lck-files found.')

    # Loop over all jobs
    job_names = list(mdb.jobs.keys())

    for job_name in job_names:
        if job_name in mdb.jobs:
            job_to_run = mdb.jobs[job_name]
            job_to_run.submit()
            job_to_run.waitForCompletion()
        else:
            print(f"Job '{job_name}' not found in MDB.")
            continue 

        # search and open odb-file
        odb_file_name = f"{job_name}.odb"
        current_directory = os.getcwd()
        full_path = os.path.join(current_directory, odb_file_name)

        if os.path.isfile(full_path):
            odb = openOdb(full_path)
            
        else:
            print(f"ODB '{odb_file_name}' not found in directory: '{current_directory}'.")
            continue

        # Initialise stress-directory for current job
        if 'E' in job_name:
            stress_directory = {dir_name: [] for dir_name in ['xx', 'yy', 'zz']}
        elif 'G' in job_name:
            stress_directory = {dir_name: [] for dir_name in ['xy', 'xz', 'yz']}
        else:
            print('Unknown kind of job')

        # Loop over directions to fill stress_directory
        for i, dir_name in enumerate(stress_directory.keys()):
            odb_element_set = odb.rootAssembly.elementSets[f'MAX_XX_ELEMENTS']
            last_step_name = odb.steps.keys()[-1]
            last_step = odb.steps[last_step_name]
            for frame in last_step.frames:
                frame_stress = []
                stress_field = frame.fieldOutputs['S']
                stress_values = stress_field.getSubset(region=odb_element_set).values
                for value in stress_values:
                    if 'E' in job_name:
                        frame_stress.append(value.data[i])
                    elif 'G' in job_name:
                        frame_stress.append(value.data[i+3])
                    else:
                        print('Unknown kind of job')

                # store mean stress value from current frame in direction i in stress directory
                stress_directory[dir_name].append(np.mean(frame_stress))

        print(f"For job {job_name} the following stress values from the odb-data are stored: : {stress_directory}")
        
        # Initialise displacement difference directory for current job
        if 'E' in job_name:
            strain_directory = {dir_name: [] for dir_name in ['xx', 'yy', 'zz']}
        elif 'G' in job_name:
            strain_directory = {dir_name: [] for dir_name in ['xy', 'xz', 'yz']}
        else:
            print('Unknown kind of job')
            strain_directory = {}
            
        # Map normal displacement components to node set names
        node_set_map = {
            'xx': ('XX', 0),  # U1
            'yy': ('YY', 1),  # U2
            'zz': ('ZZ', 2)   # U3
        }

        # Map shear components to relevant displacement directions
        shear_components_map = {
            'xy': [('XX', 0), ('YY', 1)],  # Sum of U1 and U2
            'xz': [('XX', 0), ('ZZ', 2)],  # Sum of U1 and U3
            'yz': [('YY', 1), ('ZZ', 2)]   # Sum of U2 and U3
        }

        # Loop over directions to fill strain_directory
        for i,dir_name in enumerate(strain_directory.keys()):
            if dir_name in node_set_map:
                odb_element_set = odb.rootAssembly.elementSets[f'MAX_XX_ELEMENTS']
                last_step_name = odb.steps.keys()[-1]
                last_step = odb.steps[last_step_name]
                for frame in last_step.frames:
                    frame_strain = []
                    strain_field = frame.fieldOutputs['NE']
                    strain_values = strain_field.getSubset(region=odb_element_set).values
                    for value in strain_values:
                            frame_strain.append(value.data[i])
                            
                    strain_directory[dir_name].append(np.mean(frame_strain))

            elif dir_name in shear_components_map:
                odb_element_set = odb.rootAssembly.elementSets[f'MAX_XX_ELEMENTS']
                last_step_name = odb.steps.keys()[-1]
                last_step = odb.steps[last_step_name]
                for frame in last_step.frames:
                    frame_strain = []
                    strain_field = frame.fieldOutputs['NE']
                    strain_values = strain_field.getSubset(region=odb_element_set).values
                    for value in strain_values:
                            frame_strain.append(value.data[i+3])
                        
                    # store mean stress value from current frame in direction i in strain directory
                    strain_directory[dir_name].append(np.mean(frame_strain))
                    
        print(f"For job {job_name}, the following strain values are stored: {strain_directory}")
        
        # RMSE-evaluation for current job
        rmse_weighted, mse_dict = calculate_rmse(stress_directory, strain_directory, md_data, job_name, parameters, result_dir, evaluation_count)
        rmse_list.append(rmse_weighted)
        save_mse_array(mse_dict, job_name, result_dir, evaluation_count)
        save_stress_strain(job_name, stress_directory, strain_directory, result_dir, evaluation_count)

        odb.close()

    rmse_array = np.array(rmse_list)
    total_rmse = np.sum(rmse_array)
    save_material_params(material, result_dir, evaluation_count)
    save_rmse(total_rmse, result_dir, evaluation_count)
    evaluation_count[0] += 1
    print(f"RMSE value for all jobs: {total_rmse}")
    return total_rmse

# calculate RMSE 
def calculate_rmse(stress_directory, strain_directory, md_file_dict, job_name, parameters, result_dir, evaluation_count):
    # access to md_data for given job
    if job_name not in md_file_dict:
        raise ValueError(f"Job '{job_name}' not found in md-data dictionary.")

    md_data = md_file_dict[job_name]
    mean_squared_differences = {}   
    number_of_directions = 0

    # loop over stress directions in stress_directory for given job
    for dir_name, odb_stress_values in stress_directory.items():
        if parameters.stress_analysis_direction[dir_name] == 1:
            stress_key = dir_name
            md_stress_values = md_data.data[f"stress_{stress_key}"]
            print(f'For job {job_name} in {stress_key} - direction the following stress values from the md-data are stored: {md_stress_values}')
            
            # check whether md data and odb data have the same length
            if len(odb_stress_values) != len(md_stress_values):
                print(f'The length of the odb-data values in {stress_key}-direction is {len(odb_stress_values)}')
                print(f'The length of the md-data in {stress_key}-direction is {len(md_stress_values)}')
                raise ValueError(f"odb-data array and md-data array should have the same length for direction {dir_name}.")

            # calculation of squared differences for stress
            squared_differences_stress = (np.array(odb_stress_values) - np.array(md_stress_values)) ** 2
            weights_stress = np.ones_like(squared_differences_stress)
            weights_stress[1] = 100     # weight for elastic data point
            weighted_squared_differences = squared_differences_stress * weights_stress
            mean_squared_differences_stress = np.sum(weighted_squared_differences)/np.sum(weights_stress)
            if 'E' in job_name:
                mse_stress_weight = mean_squared_differences_stress * parameters.weights['normalStress'] 
            elif 'G' in job_name:
                mse_stress_weight = mean_squared_differences_stress * parameters.weights['shearStress'] 
            
            mean_squared_differences[f'stress {dir_name}'] = mse_stress_weight
            print('Mean Squared Diff with Stress', mean_squared_differences)
            number_of_directions += 1

    # loop over strain directions in strain_directory for given job
    for dir_name, odb_strain_values in strain_directory.items():
        if parameters.strain_analysis_direction[dir_name] == 1:
            strain_key = dir_name
            md_strain_values = md_data.data[f"strain_{strain_key}"] 
            print(f'For job {job_name} in {strain_key} - direction the following strain values from the md-data are stored: {md_strain_values}')

            # check whether md data and odb data have the same length
            if len(odb_strain_values) != len(md_strain_values):
                print(f'The length of the odb-data values in {strain_key}-direction is {len(odb_strain_values)}')
                print(f'The length of the md-data in {strain_key}-direction is {len(md_strain_values)}')
                raise ValueError(f"odb-data array and md-data array should have the same length for direction {dir_name}.")
            
            print(f'For job {job_name} in {strain_key} - direction the following displacement values from the odb-data are stored: {odb_strain_values}')
            # calculation of squared differences for strain
            squared_differences_strain = (np.array(odb_strain_values) - np.array(md_strain_values)) ** 2

            weights_strain = np.ones_like(squared_differences_strain)
            weights_strain[1] = 100     # weight for elastic data point
            weighted_squared_differences_strain = squared_differences_strain * weights_strain
            mean_squared_differences_strain = np.sum(weighted_squared_differences_strain)/np.sum(weights_strain)
            if 'E' in job_name:
                mse_strain_weight = mean_squared_differences_strain * parameters.weights['normalStrain'] 
            elif 'G' in job_name:
                mse_strain_weight = mean_squared_differences_strain * parameters.weights['shearStrain'] 
            mean_squared_differences[f'strain {dir_name}'] = mse_strain_weight
            print('Mean Squared Diff with Strain', mean_squared_differences)
            print('Weights', parameters.weights)

            number_of_directions += 1

    # check if there are any directions for RMSE calculation
    if number_of_directions == 0:
        raise ValueError("No directions for RMSE calculation given.")

    # calculate RMSE
    mean_squared_diff = np.array(list(mean_squared_differences.values()))
    mean_squared_error = np.sum(mean_squared_diff) / number_of_directions  
    rmse = np.sqrt(mean_squared_error)  
    print(f"RMSE for job '{job_name}': {rmse}")
    rmse_dir_weight = 10.0
   
    for key in parameters.prescribed_directions.keys():
        if key in job_name:
            direction_weight = parameters.prescribed_directions[key]["weight"]
            rmse_dir_weight = rmse * direction_weight
            print(direction_weight)
            print(rmse_dir_weight)

    md_data_weight = parameters.md_data_dict.get(md_data.filename)
    print(md_data_weight)
    print(rmse_dir_weight)
    rmse_dir_md_weight = rmse_dir_weight * md_data_weight
    save_rmse_array(job_name, rmse, rmse_dir_weight, rmse_dir_md_weight, result_dir, evaluation_count)

    return rmse_dir_md_weight, mean_squared_differences 

# save stress and strain components
def save_stress_strain(job_name, stress_directory, strain_directory, result_directory, evaluation):
    job_directory = os.path.join(str(result_directory), job_name)
    os.makedirs(job_directory, exist_ok=True)

    components = [key.replace('stress_', '') for key in stress_directory.keys()]
    first_stress_key = next(iter(stress_directory))
    num_values = len(stress_directory[first_stress_key])
    
    # prepare data for saving
    combined_values = []
    for i in range(num_values):
        row = []
        for component in components:
            stress_key = component
            strain_key = component
            row.extend([stress_directory[stress_key][i], strain_directory[strain_key][i]])
        combined_values.append(row)
    
    headers = []
    for component in components:
        headers.extend([f'Stress_{component}', f'Strain_{component}'])
    
    # write combined data to CSV for stress and strain
    combined_file_path = os.path.join(job_directory, f"stress_strain_{evaluation}.csv")
    with open(combined_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if os.path.getsize(combined_file_path) == 0:
            writer.writerow(headers)
        for values in combined_values:
            writer.writerow(values)

    print(f"Stress and strain values for (evaluation {evaluation}) have been saved.")

# save material parameters
def save_material_params(material, result_directory, evaluation):
    material_file_path = os.path.join(result_directory, "material_parameters.csv")
    with open(material_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if os.path.getsize(material_file_path) == 0:
            writer.writerow(['Evaluation', 'YoungsModulus', 'PoissonRatio', 'PlasticYield', 'Alpha', 'Beta', 'Gamma', 'C10', 'D1'])  
        # add material parameters of curent evaluation
        writer.writerow([evaluation] + material)

    print(f"Material parameters for (evaluation {evaluation}) have been saved.")

# save RMSE 
def save_rmse(rmse, result_directory, evaluation):
    rmse_file_path = os.path.join(result_directory, "rmse.csv")
    with open(rmse_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if os.path.getsize(rmse_file_path) == 0:
            writer.writerow(['Evaluation', 'RMSE'])
        # add RMSE value of current evaluation
        writer.writerow([evaluation, rmse]) 

    print(f" RMSE for (evaluation {evaluation}) have been saved.")

# save RMSE with weights of all jobs
def save_rmse_array(job_name, rmse, rmse_dir_weight, rmse_dir_md_weight, result_directory, evaluation):
    rmse_file_path = os.path.join(result_directory, "rmse_perJob.csv")
    with open(rmse_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if os.path.getsize(rmse_file_path) == 0:
            headers = ['Evaluation', 'Job', 'RMSE', 'RMSE_DirWeight', 'RMSE_Dir_MD_Weight']
            writer.writerow(headers)
        # write job name, and RMSE values for current evaluation
        writer.writerow([evaluation, job_name, rmse, rmse_dir_weight, rmse_dir_md_weight])

    print(f"RMSEs for evaluation {evaluation} and job {job_name} have been saved.")

# save MSE values
def save_mse_array(mse_dict,job_name, result_directory, evaluation):
    job_directory = os.path.join(str(result_directory), job_name)
    os.makedirs(job_directory, exist_ok=True)
    mse_file_path = os.path.join(job_directory, "mse.csv")
    mse_keys = sorted(mse_dict.keys()) 
    mse_values = [mse_dict[key] for key in mse_keys]
    with open(mse_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if os.path.getsize(mse_file_path) == 0:
            writer.writerow(['Evaluation'] + mse_keys)
        # add MES values for current evaluation
        writer.writerow([evaluation] + mse_values)
    
    print(f"MSE values for evaluation {evaluation} have been saved.")

# copy input ile
def copy_input_parameters(src_filename, params, result_dir, index):    
    dest_filename = f"{params.test_name}_input.json"
    full_path = os.path.join(result_dir, dest_filename)
    try:
        os.makedirs(result_dir, exist_ok=True) 
        with open(src_filename, 'r') as src_file:
            parameters = json.load(src_file) 
            new_data = parameters.copy()
            # update arrays to only include the specified index 
            try:
                new_data['PlasticYield']['value'] = parameters['PlasticYield']['value'][index]
                new_data['Alpha']['value'] = parameters['Alpha']['value'][index]
                new_data['Beta']['value'] = parameters['Beta']['value'][index]
                new_data['Gamma']['value'] = parameters['Gamma']['value'][index]
                new_data['YoungsModulus']['value'] = parameters['YoungsModulus']['value'][index]
                new_data['PoissonRatio']['value'] = parameters['PoissonRatio']['value'][index]
            except IndexError:
                print(f"Index {index} is out of range for one of the parameter arrays.")
                return
        
        with open(full_path, 'w') as dest_file:
            json.dump(parameters, dest_file, indent=4) 
        
        print(f"Copied input parameters in '{full_path}'.")
    
    except Exception as e:
        print(f"Error when copying input parameters: {e}")
    
    return None 

# copy reference data
def copy_md_file(src_filename, job_name, result_directory, input_directory):
    job_directory = os.path.join(str(result_directory), job_name)
    os.makedirs(job_directory, exist_ok=True)
    dest_filename = os.path.join(job_directory, os.path.basename(src_filename))
    src_filepath = os.path.join(input_directory, src_filename)
    try:
        with open(src_filepath, 'rb') as src_file:  
            with open(dest_filename, 'wb') as dest_file:  
                dest_file.write(src_file.read())
        
        print(f"Die Datei '{src_filename}' wurde erfolgreich nach '{dest_filename}' kopiert.")

    except Exception as e:
        print(f"Fehler beim Kopieren der Datei: {e}")

    return None
        
# svae ODB
def save_odb():
    # loop over all jobs in job_names
    job_names = list(mdb.jobs.keys())

    for job_name in job_names:
        if job_name in mdb.jobs:
            job_to_run = mdb.jobs[job_name]
            job_to_run.submit()
            job_to_run.waitForCompletion()
        else:
            print(f"Job '{job_name}' not found in MDB.")
            continue 

        odb_file_name = f"{job_name}.odb"
        current_directory = os.getcwd()
        full_path = os.path.join(current_directory, odb_file_name)
        if os.path.isfile(full_path):
            odb = openOdb(full_path)   
        else:
            print(f"ODB '{odb_file_name}' not found in directory: '{current_directory}'.")
            continue 
        odb.save()
        odb.close()
        print(f"Odb for job {job_name} has been saved in {current_directory}.")
        
    return None

# write scipy.minize message
def write_rmse_message(RMSE, output_filename, directory):
    full_path = os.path.join(directory, output_filename)
    with open(full_path, 'w') as file:
        file.write(f"Optimization result:\n")
        file.write(f"Final RMSE value: {RMSE.fun}\n")
        file.write(f"Optimized parameters: {RMSE.x}\n")
        file.write(f"Optimization success: {RMSE.success}\n")
        file.write(f"Message: {RMSE.message}\n")
        if 'allvecs' in RMSE:
            file.write(f"All optimization paths: {RMSE.allvecs}\n")

    print(f"RMSE results have been saved to {output_filename}")

# main function
def main():

    print('###\n')
    print('NEW SCRIPT RUN\n')
    print('###\n')

    top_dir = '/calculate/Ziegler/Abaqus_Scripting/MultiJobAnalysis'
    result_directory = '/calculate/Ziegler/Abaqus_Scripting/MultiJobAnalysis/PA_Results'
    input_directory = os.path.join(top_dir, 'PA_Input')
    parameter_input = "AbaqusInput.json"
    parameter_input_path = os.path.join(input_directory, parameter_input)
    cube_parameters = read_parameter_file(parameter_input_path, input_directory)
    print(cube_parameters)

    # loop over all initial value combinations
    for i in range(cube_parameters.number_of_params): 
        mdb = Mdb()
        work_directory_name = f"{cube_parameters.test_name}_{i}"
        work_directory = os.path.join(result_directory, work_directory_name)
        os.makedirs(work_directory, exist_ok=True)
        model_directory = os.path.join(work_directory, cube_parameters.model_name)
        os.makedirs(model_directory, exist_ok = True)
        os.chdir(model_directory)
        copy_input_parameters(parameter_input_path, cube_parameters, work_directory, i)
        mdb_cubes_dict = {}  
        md_data_dict = {}
        # create Abaqus model for all load parameters and load cases
        for filename in cube_parameters.md_data_dict.keys():
            for direction, properties in cube_parameters.prescribed_directions.items():
                if properties['active']  != 0:
                    md_data_path = os.path.join(input_directory, filename)
                    mdb_cube = MDBCube(cube_parameters,md_data_path, model_directory, direction)  
                    mdb_cube.create_cube(i)
                    job_name = mdb_cube.create_job(filename, direction)
                    mdb_cube.update_boundary_condition(direction)
                    mdb_cube.update_increment_size()
                    mdb_cube.save_mdb()
                    mdb_cubes_dict[filename] = mdb_cube 
                    md_data = MDData(filename, input_directory)
                    md_data_dict[job_name] = md_data
                    copy_md_file(filename, job_name, work_directory, input_directory)
            
        print('Result directory', work_directory)
        
        lck_files = glob.glob('*.lck')
        if lck_files:
            for file in lck_files:
                try:
                    os.remove(file)
                except Exception as e:
                    print(f'Error when deleting {file}: {e}')
        else:
            print('No .lck-file found.')
        
        # scaled material parameter boundaries
        bounds =[(0,1), (0,1), (0,1), (0,1)] 
        scaled_material = [cube_parameters.plastic_yield_scaled[i],
        cube_parameters.alpha_scaled[i], cube_parameters.beta_scaled[i], cube_parameters.gamma_scaled[i]] 
    
        evaluation_count = [0]
        # call scipy.minize()
        RMSE = minimize(optimization_multiple_stresses, scaled_material, args=(cube_parameters,  md_data_dict, work_directory, evaluation_count),
            method='nelder-mead', bounds = bounds, options={'disp': True, 'return_all': True, 'maxiter': cube_parameters.iteration_number})
        
        print(RMSE)
        write_rmse_message(RMSE, 'rmse_message.txt', work_directory)
        save_odb()
        mdb.save()
        mdb.close()

if __name__=="__main__":
    main()





