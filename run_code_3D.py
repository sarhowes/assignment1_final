"""
    ====== Main script to run all code ======
            ------- 3D version -------
      created by: Sarah Howes, Lars Reems
    =========================================  

Run this script to perform the main argon dynamics simulation in 3D.
Using user inputs prompts, you can customize the setup for the simulation.
Or, you can choose the default values for the simulation 
and one of the three phases of matters listed below:
###################################################
Different states of matter that have been tested:
# gas: density = 0.3, temperature = 3.0
# liquid: density = 0.8, temperature = 1.0
# solid: density = 1.2, temperature = 0.5
###################################################

This script imports the class Particle_Functions3D from definitions_3D.py
to simulate the positions and velocities of argon atoms within a volume
with periodic boundaries.

This script also will create new folders for each state and run number
in order to save the outputted data and plots that are produced.

You can customize which plots and values are calculated/saved using 
user input prompts that are provided after running the script.

"""

import numpy as np
import matplotlib.pyplot as plt
from definitions_3D import Particle_Functions3D
import os
from tqdm import tqdm
import glob
import pandas as pd



def main_simulation(number_particles:float, length:float, number_steps:float, run:float, cut_off:float):
    """
    Main function for running the 3D simulation through given number of time steps. Uses functions from
    Particle_Functions class to find the next positions and velocities for each time step, saves full time
    sequence to a numpy ndarray.

    Args:
        number_particles (float): number of particles in simulation
        length (float): length of box
        number_steps (float): number of time steps
        run (float): run number, for making new files
        cut_off (float): fraction of steps to rescale velocities

    Returns:
        np.ndarray: shape (number_steps, number_particles, (x,y,vx,vy)) all positions and velocities for
        each particle for each time step
    """
    #Initialize coordinates and velocities for 10 particles
    diff = length/(number_cubes)**(1/3)
    coordinate_base = list(np.linspace(diff/4,length-diff/4,int(2*number_cubes**(1/3))))
    initial_x_coord = coordinate_base*round(number_cubes**(2/3))*2

    initial_y_coord = []
    for i in range(round(2*(number_cubes**(1/3)))):
        for j in range(int((number_cubes)**(1/3))):
            count = 0
            while count < int((number_cubes)**(1/3)*2):
                if i%2==0:
                    if count%2==0:
                        initial_y_coord.append(coordinate_base[2*j])
                    else:
                        initial_y_coord.append(coordinate_base[2*j+1])
                else:
                    if count%2==0:
                        initial_y_coord.append(coordinate_base[2*j+1])
                    else:
                        initial_y_coord.append(coordinate_base[2*j])
                count += 1

    initial_z_coord = []
    for i in range(len(coordinate_base)):
        count = 0
        while count< round((number_cubes**(2/3)*2)):
            initial_z_coord.append(coordinate_base[i])
            count += 1

    initial_velocity_x = (np.random.normal(0,np.sqrt(temperature),number_particles))
    initial_velocity_y = (np.random.normal(0,np.sqrt(temperature),number_particles))
    initial_velocity_z = (np.random.normal(0,np.sqrt(temperature),number_particles))
                        
    # create 4xnumber_particles sized array
    dtypes = {'names':['x_coord', 'y_coord', 'z_coord', 
                       'velocity_x', 'velocity_y', 'velocity_z'], 
              'formats':[float,float,float,float,float,float]}
    
    particles = np.zeros_like(initial_x_coord,dtype=dtypes)
    
    for i in range(len(initial_x_coord)):
        particles[i]= initial_x_coord[i],initial_y_coord[i],initial_z_coord[i],\
            initial_velocity_x[i],initial_velocity_y[i],initial_velocity_z[i]


    # create number_stepsx(particle_array) sized array
    i_stored_particles = np.zeros_like(particles, dtype=dtypes)
    stored_particles = []
    for i in range(number_steps):
        stored_particles.append(i_stored_particles)

    stored_particles = np.array(stored_particles)

    #add initial array to stored_particles
    stored_particles[0] = particles

    # calculate next array based on previous array
    for t in tqdm(range(number_steps-1)):

        # compute next positions
        x_new, y_new, z_new = particle_simulator.new_positions(stored_particles[t], time_step=time_step)

        # initiate new data array
        new_data = np.zeros_like(x_new,dtype=dtypes)
        # append new positions to new data array
        for i in range(len(x_new)):
            new_data[i]= x_new[i],y_new[i],z_new[i],0,0,0

        # compute next velocities (using old and new positions)
        vx_new, vy_new, vz_new = particle_simulator.new_verlet_velocity(stored_particles[t], 
                                                             new_data=new_data, time_step=time_step)

        # adjust the velocities with rescaling parameter for first 10% of steps
        if t <= cut_off:
            v_tot = np.sqrt(vx_new**2 + vy_new**2 + vz_new**2)
            v_tot_squared_sum = np.sum(v_tot**2)
            rescaling_parameter = np.sqrt(((number_particles-1)*3*temperature)/(v_tot_squared_sum))
            vx_new = rescaling_parameter*vx_new
            vy_new = rescaling_parameter*vy_new
            vz_new = rescaling_parameter*vz_new

        # update new data array, append new vx, vy
        for i in range(len(x_new)):
            new_data[i]= x_new[i],y_new[i],z_new[i],vx_new[i],vy_new[i],vz_new[i]
        # update all data
        stored_particles[t+1] = new_data
    # save data
    np.save(f'{state}/run3D_{run}/all_data.npy', stored_particles)
    return stored_particles



def store_pressure_one_run(stored_particles:np.ndarray, cut_off:float, run_number:int):
    """Calculate and save total 3D pressure for the system

    Args:
        stored_particles (np.ndarray): full simulation data with all time steps
        cut_off (float): fraction of steps to exclude so to only consider equilibrium values
        run_number (int): the run number of the simulation, for saving the pressure value

    Returns:
        float: total pressure of the system
    """

    path = f'{state}/pressure_3D'
    if not os.path.exists(path): 
        os.makedirs(path)

    #adding up variables for all timesteps to get pressure
    time_summed_for_P = []
    for t in tqdm(range(number_steps-1)):
        #calculate the pressure of the system
        if t > cut_off:
            rvals, diffs = particle_simulator.distance_values_one_timestep(stored_particles[t])
            np.fill_diagonal(rvals, 0.0)
            dU_vals = particle_simulator.lennard_jones_potential_derivative(stored_particles[t])
            each_particle_sum = np.sum(rvals*dU_vals, axis=1)
            summed = np.sum(each_particle_sum)/2 # divide by 2 for repeated calculation
            time_summed_for_P.append(summed)

    time_summed_for_P = np.array(time_summed_for_P)

    # calculate pressure
    pressure = (1 - (1 / (6*number_particles*temperature) )*\
                     np.mean(time_summed_for_P))*density*temperature #unitless
    print(f'Pressure: {pressure}')
    print(f'Saving pressure to: {path}')
    np.save(f'{path}/pressure_run_{run_number}.npy', float(pressure))
    return pressure



def store_plot_PCF_one_run(stored_particles:np.ndarray, run:float, cut_off:float):
    """Calculate, plot, and save the 3D pair correlation function values

    Args:
        stored_particles (np.ndarray): full simulation data with all time steps
        run (float): run number to save to folder
        cut_off (float): fraction of steps to exclude so to only consider equilibrium values
    
    Returns:
        np.array: bin values for pair correlation function
    """
    path = f'{state}/pair_correlation_function_3D'
    if not os.path.exists(path): 
        os.makedirs(path)

    #adding up bins for all timesteps
    number_bins = 100
    bins_data_summed = np.zeros(number_bins)

    for t in tqdm(range(number_steps-1)):
        # calculate the histogram for pair correlationbins
        if t > cut_off:
            bins_data, bins = particle_simulator.pair_correlation_histogram(stored_particles[t])
            bins_data_summed += bins_data

    # calculate pair correlation function
    bins_data_summed = (bins_data_summed / (number_steps-cut_off)) / 2 #/2 because double counting of pairs
    bin_size = (length/2) / number_bins
    pair_correlation_function = ((2*length**3) / (number_particles*(number_particles-1))) * (bins_data_summed/(4*np.pi*(bins**2)*bin_size))

    # plot pair correlation function
    width = bins[1] - bins[0]
    plt.bar(bins, height=pair_correlation_function, width=width, align='edge', color='green', edgecolor='k')
    plt.xlabel('Distance')
    plt.ylabel('Counts')
    plt.title('Pair Correlation Function')
    plt.savefig(f'{state}/run3D_{run}/pair_correlation_function.png',dpi=300)
    plt.close()

    print('saving pair correlation function to:', path)
    np.save(f'{path}/pcf_run_{run}.npy', pair_correlation_function)
    return pair_correlation_function



def plot_kinetic_potential_energy(stored_particles:np.ndarray, run:float, cut_off:float):
    """Calculate, plot, and save the 3D total potential and kinetic energies of the simulation
    over all time steps.

    Args:
        stored_particles (np.ndarray): full simulation data with all time steps
        run (float): run number to save to folder
        cut_off (float): fraction of steps to exclude so to only consider equilibrium values
    """
    # calculate total potential and kinetic energy for system for each time step
    Ekin_tot, Epot_tot = particle_simulator.sum_kinetic_potential_energy(stored_particles, cut_off=cut_off)
    #normalize to initial total energy
    E_tot = Ekin_tot+Epot_tot


    # convert time to SI units for plotting
    mass_argon = 6.6335209e-26 #kg
    sigma = 3.405e-10 #m
    kb = 1.380649e-23 # kg m2 s-2 K-1
    epsilon = 119.8*kb # kg m2 s-2  -> joules
    epsilon_eV  = epsilon * 6.24150907e18 #eV

    timestep_values = np.arange(0,len(E_tot), 1)
    timestep_SI = time_step*timestep_values*np.sqrt(mass_argon*(sigma**2)/epsilon) # seconds
    timestep_SI = timestep_SI *1e12 # picoseconds


    plt.grid()
    plt.plot(timestep_SI, E_tot*epsilon_eV, c='k', marker='.', alpha=0.5,label='Both')
    plt.plot(timestep_SI, Ekin_tot*epsilon_eV, c='r', marker='.', alpha=0.5,label='Kinetic')
    plt.plot(timestep_SI, Epot_tot*epsilon_eV, c='b', marker='.', alpha=0.5,label='Potential')
    plt.legend()
    plt.xlabel('Time (ps)')
    plt.ylabel('Total Energy of Whole System (eV)')
    plt.title(f'Total Energy for {state} phase')
    plt.savefig(f'{state}/run3D_{run}/total_energy.png',dpi=300)
    plt.close() 



def plot_3D_positions_one_timestep(stored_particles:np.ndarray, plot_index:int):
    """Plot one 3D position figure of simulation.

    Args:
        stored_particles (np.ndarray): full simulation data with all time steps
        plot_index (int): time step (index) of full simulation to plot in 3D
    """

    import plotly.express as px
    # creating figure
    one_step = stored_particles[plot_index]
    xvals = one_step['x_coord']
    yvals = one_step['y_coord']
    zvals = one_step['z_coord']
    vals = {'x_coord': xvals,
            'y_coord': yvals,
            'z_coord': zvals}
    df = pd.DataFrame(vals)
    fig = px.scatter_3d(df, x='x_coord', y='y_coord', z='z_coord')
    ### end if statement
    fig.update_layout(
        scene = dict(
            xaxis = dict(range=[0,particle_simulator.length],),
                        yaxis = dict(range=[0,particle_simulator.length],),
                        zaxis = dict(range=[0,particle_simulator.length],),))
    fig.show()



def mean_pressure_all_runs(pressures_path:str):
    """Calculate the mean pressure and error of 3D simulations
    over a series of separate runs.

    Args:
        pressures_path (str): path to folder containing all saved pressure values

    Returns:
        float, float: mean and standard deviation of saved pressures
    """

    pressure_files = os.listdir(pressures_path)
    kb = 1.380649e-23 # kg m2 s-2 K-1
    epsilon = 119.8*kb # kg m2 s-2
    sigma = 3.405e-10 # m
    
    pressures_si = []
    pressures_nounit =[]
    for f in pressure_files:
        pres = np.load(f'{pressures_path}/{f}')
        pressures_si.append(pres*(epsilon/(sigma**3)))
        pressures_nounit.append(pres)

    print('mean and std pressures no units:', np.mean(pressures_nounit), np.std(pressures_nounit))
    print('mean and std pressures SI units', np.mean(pressures_si), np.std(pressures_si))

    print(f'saving mean pressure and error')
    np.save(f'{state}/mean_pressure_3D.npy', np.mean(pressures_nounit))
    np.save(f'{state}/mean_pressure_error_3D.npy', np.std(pressures_nounit))

    return np.mean(pressures_si), np.std(pressures_si)



def mean_PCF_all_runs(pcf_path:str):
    """Calculate and plot the mean and standard deviation for 
    pair correlation function for a series of separate 3D simulation runs.

    Args:
        pcf_path (str): path to saved pair correlation function values

    Returns:
        np.array, np.array: the mean and error values for each PCF bin
    """
    pcf_files = os.listdir(pcf_path)
    all_files = []
    for file in pcf_files:
        pcf = np.load(f'{pcf_path}/{file}')
        all_files.append(pcf)
    
    all_files = np.array(all_files)
    mean_pcf = np.mean(all_files, axis=0)
    stdev_pcf = np.std(all_files, axis=0)
    np.save(f'{state}/mean_pcf_3D.npy', mean_pcf)
    np.save(f'{state}/mean_pcf_std_3D.npy', stdev_pcf)
    
    # plot pair correlation function
    bins = np.linspace(0,particle_simulator.length/2,100)
    sigma = 3.405 # angstroms

    bins_si = bins*sigma
    width = bins_si[1]-bins_si[0]
    if state == 'solid':
        color = 'forestgreen'
        edgecolor = 'darkgreen'
    elif state== 'liquid':
        color = 'royalblue'
        edgecolor = 'mediumblue'
    elif state == 'gas':
        color = 'chocolate'
        edgecolor = 'saddlebrown'
    else:
        color = 'firebrick'
        edgecolor = 'maroon'
    plt.bar(bins_si, height=mean_pcf, width=width, align='edge', edgecolor=edgecolor, color=color)
    plt.xlabel('Distance (Å)')
    plt.ylabel('Counts')
    plt.title(f'Mean pair correlation function for {state} phase')
    plt.savefig(f'{state}/mean_pair_correlation_function_3D.png',dpi=300)
    plt.close()

    return mean_pcf, stdev_pcf



def define_temp_density():
    """User input to define temperature and density values for desired simulated phase of argon

    Returns:
        float, float, str: temperature and density values for simulation, phase label for making new files
    """
    state = input('>> Choose desired state (solid/liquid/gas/custom):')
    if state == 'solid':
        density = 1.2 #sigma^-3
        temperature = 0.5 #epsilon/kb
    elif state == 'liquid':
        density = 0.8 #sigma^-3
        temperature = 1.0 #epsilon/kb
    elif state == 'gas':
        density = 0.3 #sigma^-3
        temperature = 3.0 #epsilon/kb
    elif state == 'custom':
        density = float(input('>> Input density (units: sigma^-3):'))
        temperature = float(input('>> Input temperature (units: epsilon/kb):'))
        state = f'custom_den_{density}_temp_{temperature}'
    else:
        print('Unknown input')
        exit()

    if not os.path.exists(state): 
        print('Creating new folder for state:', state)
        os.makedirs(state)
    print('Using temperature, density:', temperature,',',density)
    return temperature, density, state



def simulation_setup():
    """Set up user inputs that will customize what each run will produce

    Returns:
        list of floats: returns all the needed variables to set up the simulation, including: 
        the number of cubes the makes up the box, the number of particles, length of each box side, 
        time step (h), number of steps, cut off for equilibrium tuning, temperature, density, and 
        state/phase of system
    """

    print(
        '===========================================================\n'
        '======== Molecular Dynamics with the Argon Atom ===========\n'
        '====== simulation by: Sarah Howes and Lars Reems ==========\n'
        '==========================================================='
        )


    print(
        '===== Default simulation values: =====\n'
        '* number_cubes_one_direction: 3\n'
        '* timestep (units: sqrt(m*sigma^2/epsilon)): 0.001\n'
        '* number_steps: 1000\n'
        '* tuning_percent: 0.20\n'
        '======================================'
        )
    custom_setup = input('>> Would you like to customize the simulation setup? (y/n):')
    if custom_setup == 'y':
        number_cubes_one_direction = int(input('>> Input number of cubes to simulate in \none direction (number will be cubed):'))
        time_step = float(input('>> Input time step \n(time in units sqrt(m*sigma^2/epsilon)):'))
        number_steps = int(input('>> Input number of steps:'))
        tuning_percent = float(input('>> Input fraction of simulation to run equilibrium on:'))
    elif custom_setup == 'n':
        number_cubes_one_direction = 3        
        time_step = 0.001
        number_steps = 1000
        tuning_percent = 0.20
    else:
        print('Unknown command')
        exit()

    number_cubes = number_cubes_one_direction**3
    number_particles = number_cubes*4 
    print('Number of particles for this simulation: ', number_particles)
    cut_off = tuning_percent*number_steps
    temperature, density, state = define_temp_density()
    length = (number_particles / density)**(1/3) #for 3D

    print(
        '==================================\n'
        '==== Final setup parameters: =====\n'
        f'* number cubes: {number_cubes}\n'
        f'* number particles: {number_particles}\n'
        f'* length (sigma): {length}\n'
        f'* time_step (units: sqrt(m*sigma^2/epsilon)): {time_step}\n'
        f'* number steps: {number_steps}\n'
        f'* tuning percent: {tuning_percent}\n'
        f'* temperature (units: epsilon/kb): {temperature}\n'
        f'* density (units: sigma^-3): {density}\n'
        f'* state: {state}\n'
        '=================================='
    )

    proceed = input('>> Proceed with setup? (y/n):')

    if proceed == 'n':
        exit()
    elif proceed == 'y':
        pass
    else:
        print('Unknown command')
        exit()

    return  number_cubes, number_particles, length, time_step, number_steps, cut_off, temperature, density, state



def choose_what_to_run(state:str):
    """Select which functions from the above definitions that will run during the simulation.
    You first select the number of runs, then select which of the following you want to be produced:
        - calculate/store pressure of run
        - plot energy diagram for run
        - plot pair correlation function for run
        - plot a 3D diagram of position of particles for one time step in one run
        - calculate the mean pressure over a series of runs
        - calculate/plot the mean pair correlation function over a series of runs

    Args:
        state (str): state of the system. Used to save files to folder

    Returns:
        list of strings: strings are either `y` or `n`, indicating whether or not to run a particular function
    """

    number_runs = int(input('>> Input number of separate simulations to run:'))

    print(
        '====================================\n'
        '===== Default commands to run: =====\n'
        '* main_simulation (required): on\n'
        '* store_pressure (per run): on\n'
        '* plot_energy (per run): on\n'
        '* plot_pair_correlation_function (per run): on\n'
        '* plot_one_step_3D_position (per run): off\n'
        '* mean_pressure (average all runs): on\n'
        '* mean_pair_correlation_function (average all runs): on\n'
        '===================================='
    )

    custom_commands = input('>> Would you like to customize? (y/n):')
    if custom_commands == 'y':
        run_store_pressure = input('store_pressure (on/off):')
        run_plot_energy = input('plot_energy (on/off):')
        run_plot_pcf = input('plot_pair_correlation_function (on/off):')
        run_plot_3D_position = input('plot_one_step_3D_position (on/off):')
        run_mean_pressure = input('mean_pressure (on/off):')
        run_mean_pcf = input('mean_pair_correlation_function (on/off):')
    elif custom_commands == 'n':
        run_store_pressure, run_plot_energy, run_plot_pcf, run_mean_pressure, run_mean_pcf = ['on']*5
        run_plot_3D_position = 'off'
    else:
        print('Unknown command')
        exit()    
    
    print(
        '=======================================\n'
        '======= Chosen commands to run: =======\n'
        f'* Number of simulations to run: {number_runs}\n'
        '* main_simulation (required): on\n'
        f'* store_pressure (per run): {run_store_pressure}\n'
        f'* plot_energy (per run): {run_plot_energy}\n'
        f'* plot_pair_correlation_function (per run): {run_plot_pcf}\n'
        f'* plot_one_step_3D_position (per run): {run_plot_3D_position}\n'
        f'* mean_pressure (average all runs): {run_mean_pressure}\n'
        f'* mean_pair_correlation_function (average all runs): {run_mean_pcf}\n'
        '======================================='
    )

    proceed = input('>> Start simulation? (y/n):')

    if proceed == 'n':
        exit()
    else:
        pass

    return number_runs, run_store_pressure, run_plot_energy, run_plot_pcf, run_mean_pressure, run_mean_pcf, run_plot_3D_position



if __name__ == "__main__":

    ## setting up for simulation
    number_cubes, number_particles, length, \
        time_step, number_steps, cut_off, \
        temperature, density, state = simulation_setup()
    
    number_runs, run_store_pressure, \
        run_plot_energy, run_plot_pcf, \
        run_mean_pressure, run_mean_pcf, run_plot_3D_position = choose_what_to_run(state)

    # start particle simulator
    particle_simulator = Particle_Functions3D(length=length)
    print('/=/=/=/=/ STARTING SIMULATION /=/=/=/=/')
    for i in range(number_runs):
        run_number = i+1
        print('=========== RUN', run_number, '=============')
        path = f'{state}/run3D_{run_number}'
        if not os.path.exists(path): 
            os.makedirs(path)


        print('Running main simulation...')
        stored_particles = main_simulation(number_particles, length, number_steps, run=run_number, cut_off=cut_off)
        # stored_particles = np.load(f'{path}/all_data.npy') # uncomment this and comment out above line only if 
                                                             # you have previously saved data that you want to plot

        if run_store_pressure == 'on':
            print('Finding pressure...')
            pressure = store_pressure_one_run(stored_particles, cut_off=cut_off, run_number=run_number)

        if run_plot_energy == 'on':
            print('Plotting energy...')
            plot_kinetic_potential_energy(stored_particles, run=run_number, cut_off=cut_off)

        if run_plot_pcf == 'on':
            print('plotting pair correlation function...')
            store_plot_PCF_one_run(stored_particles, run=run_number, cut_off=cut_off)

        if run_plot_3D_position == 'on':
            print('plotting positions...')
            plot_3D_positions_one_timestep(stored_particles, plot_index=0)
    
    print('========== RUNS FINISHED ==========')
    if run_mean_pressure == 'on':
        print(f'Calculating mean pressure over {number_runs} runs...')
        mean_pressure_all_runs(f'{state}/pressure_3D')
    if run_mean_pcf == 'on':
        print(f'Plotting mean PCF over {number_runs} runs...')
        mean_PCF_all_runs(f'{state}/pair_correlation_function_3D')

