"""
    ====== Class script for running the simulations ======
              created by: Sarah Howes, Lars Reems
    ======================================================

Includes all the physics formulas that are used in calculating
the new positions and velocities of a 3D system of argon atoms within 
a volume with periodic boundaries.

This script is imported into run_code_3D.py in order to perform
all simulations. 

"""
import numpy as np


class Particle_Functions3D:

    def __init__(self, length):
        self.length = length

        
    def distance_values_one_timestep(self, data:np.ndarray):
        """Calculate distances and x,y,z-coordinate distances between every particle in one time step

        Args:
            data (np.ndarray): data array for one time step

        Returns:
            np.ndarray, np.ndarray: distances of shape (number_particles, number_particles)
                                    differences of shape (number_particles, number_particles, (x_coord,y_coord,z_coord))
        """
        length = self.length
        x_coordinate, y_coordinate, z_coordinate = data['x_coord'], data['y_coord'], data['z_coord']
        positions = np.array([x_coordinate,y_coordinate,z_coordinate]).T

        differences = (positions[:,np.newaxis,:]-positions[np.newaxis,:,:] + length/2) % length - length/2

        squared_distances = np.sum(differences**2, axis=2)

        distances = np.sqrt(squared_distances)
        np.fill_diagonal(distances, np.inf)

        return distances, differences



    def lennard_jones_potential(self, data:np.ndarray):
        """calculate total Lennard-Jones potential energy for one time step

        Args:
            data (np.ndarray): current time step data

        Returns:
            float: total potential energy
        """
        distance_values, difference_values = self.distance_values_one_timestep(data)
        U = 4 * ((1/distance_values)**12 - (1/distance_values)**6)
        U_sum = np.sum(U)/2
        return U_sum



    def lennard_jones_potential_derivative(self, data:np.ndarray):
        """calculate derivative of total Lennard-Jones potential energy for one time step

        Args:
            data (np.ndarray): current time step data

        Returns:
            float: derivative of total potential energy
        """
        distance_values, difference_values = self.distance_values_one_timestep(data)
        dU = 4 * (-12*(1/distance_values)**13 + 6*(1/distance_values)**7)  # r in units sigma, E in units epsilon

        return dU



    def force_between_atoms(self, data:np.ndarray):
        """calculate force one particle feels for every other particle

        Args:
            data (np.ndarray): current time step data

        Returns:
            float, float: force in x and y direction
        """
        distance_values, difference_values = self.distance_values_one_timestep(data)
        dU = self.lennard_jones_potential_derivative(data)
        force_x = (-dU/distance_values) * difference_values[:,:,0] #dx
        force_y = (-dU/distance_values) * difference_values[:,:,1] #dy
        force_z = (-dU/distance_values) * difference_values[:,:,2] #dz


        force_x_sum = np.sum(force_x, axis=1)
        force_y_sum = np.sum(force_y, axis=1)
        force_z_sum = np.sum(force_z, axis=1)
                
        return force_x_sum, force_y_sum, force_z_sum



    def new_verlet_velocity(self, data:np.ndarray, new_data:np.ndarray, time_step:float):
        """calculate new velocity of particle in x and y direction using the Verlet algorithm

        Args:
            data (np.ndarray): current time step data
            new_data (np.ndarray): new time step data
            time_step (float): size of time step

        Returns:
            float, float: new x and y velocity
        """

        # find force for previous time step data
        force_x,force_y,force_z = self.force_between_atoms(data)
        # find force for current time step data
        force_xnew, force_ynew, force_znew = self.force_between_atoms(new_data)

        velocity_x, velocity_y, velocity_z = data['velocity_x'], data['velocity_y'], data['velocity_z']

        #Calculate the new x,y velocities
        velocity_x_new = velocity_x + 0.5*time_step*(force_x + force_xnew)
        velocity_y_new = velocity_y + 0.5*time_step*(force_y + force_ynew)
        velocity_z_new = velocity_z + 0.5*time_step*(force_z + force_znew)


        return velocity_x_new, velocity_y_new, velocity_z_new



    def new_positions(self, data:np.ndarray, time_step:float):
        """calculate the new position of a particle in the x and y direction
        time step: units sqrt(m*sigma**2/epsilon)
        velocity: units sqrt(epsilon/m)
        position: units sigma


        Args:
            data (np.ndarray): data array for one time step
            time_step (float): size of time step

        Returns:
            float, float: new position of particle in x and y direction
        """

        # find force for previous time step data
        force_x,force_y,force_z = self.force_between_atoms(data)

        data_x_coord, data_y_coord, data_z_coord = data['x_coord'], data['y_coord'], data['z_coord']
        data_velocity_x, data_velocity_y, data_velocity_z = data['velocity_x'], data['velocity_y'], data['velocity_z']


        # calculate new x,y position using previous velocity and force
        x_coord_new = data_x_coord + data_velocity_x*time_step + 0.5*(time_step**2)*force_x
        x_coord_new = x_coord_new%self.length

        y_coord_new = data_y_coord + data_velocity_y*time_step + 0.5*(time_step**2)*force_y
        y_coord_new = y_coord_new%self.length

        z_coord_new = data_z_coord + data_velocity_z*time_step + 0.5*(time_step**2)*force_z
        z_coord_new = z_coord_new%self.length

        return x_coord_new, y_coord_new, z_coord_new # units sigma



    def sum_kinetic_potential_energy(self, stored_particles:np.ndarray, cut_off:float):
        """calculate total kinetic and potential energy for each time step

        Args:
            stored_particles (np.ndarray): array of all particles at all time steps
            cut_off (float): fraction of first data values that are ignored (tuning equilibrium)

        Returns:
            np.array, np.array: total kinetic and potential energy for system at each time step
        """
        # total energy plot over time
        t_step = 0
        Ekin_tot = []
        Epot_tot = []

        # loop over each time step
        for one_step in stored_particles:
            # only calculate energies after equilibrium has reached
            if t_step > cut_off:
                total_velocity = np.sqrt(one_step['velocity_x']**2 + one_step['velocity_y']**2 + one_step['velocity_z']**2)
                Ekins = np.sum(0.5*(total_velocity**2))
                Epots = self.lennard_jones_potential(one_step)
                Ekin_tot.append(Ekins)
                Epot_tot.append(Epots)
            t_step+=1

        Ekin_tot = np.array(Ekin_tot)
        Epot_tot = np.array(Epot_tot)
        
        return Ekin_tot, Epot_tot



    def pair_correlation_histogram(self, data:np.ndarray):
        """calculate the histogram bin values for the pair correlation function

        Args:
            data (np.ndarray): current time step data

        Returns:
            np.array, np.array: counts per bin, bins
        """
        length = self.length
        bins = np.linspace(0,length/2,101)

        # find all distances for one timestep
        distance_values = self.distance_values_one_timestep(data)[0]
        np.fill_diagonal(distance_values, 0.0)
        distance_values_flattened = distance_values.flatten().flatten()

        # exclude distances between same point
        distance_values_flattened = distance_values_flattened[distance_values_flattened != 0]

        # create histogram
        bins_data = np.histogram(distance_values_flattened, bins)[0]
        return bins_data, bins[1:]