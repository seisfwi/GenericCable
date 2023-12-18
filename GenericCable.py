# Author: Haipeng Li
# Affiliate: Stanford University
# Email: haipeng@sep.stanford.edu 


class GenericCable(object):
    """ Generic cable class for Distributed Acoustic Sensing (DAS) data 
    modeling (forward) and inversion (adjoint).

    Parameters
    ----------
    traj : ndarray
        The trajectory in the Cartesian coordinate system with shape (npts, 3),
        where npts >= 2.
    chann_len : float
        The length of the DAS channel.
    chann_num : int
        The number of DAS channels.
    gauge_len : float
        The length of the gauge.
    chann_beg : float, optional
        The first channel in arc length. If not specified, it is set to be 
        gauge_len / 2.0, which means that the first channel starts at very 
        beginning of the cable, covering range [0, gauge_len] in arc length.
    threshold : float, optional
        The distance threshold for removing duplicated points. The default is 0.1, 
        which means that if two neighboring points are closer than 0.1, they will
        considered to be the same. For example, if DAS required two close receivers, 
        e.g., [100.0, 200.0] m and [100.05, 200.0] m, they will be treated as 
        the same one and only inquire the wavefiled at [100.0, 200.0] m.

    Notes
    -----
        1. The trajectory must be a 3D array with shape (npts, 3), where npts >= 2 and 
           each row is in the form of [x, y, z]. The trajectory can be either 2D or 3D.
           For 2-D case, the y coordinates must be all zeros, i.e., traj[:, 1] = 0.0.
        2. The channel length is not necessarily equal to the gauge length.
        3. The specified channel number should make sure that the last channel ends
           at the end of the cable, i.e., chann_beg + chann_len * chann_num <= traj_len.
        4. The distance threshold is related to the grid size of the forward modeling.
           For example, if the grid size is 10 m, a threshold of 0.1 m may be reasonable.
    """

    def __init__(self, traj: np.ndarray, chann_len: float, chann_num: int, 
                gauge_len: float, chann_beg: float = None, 
                threshold: float = 0.1) -> None:

        self.traj = traj
        self.chann_len = chann_len
        self.chann_num = chann_num
        self.gauge_len = gauge_len
        self.threshold = threshold
        self.chann_beg = chann_beg if chann_beg is not None else gauge_len / 2.0
        self.chann_end = self.chann_beg + self.chann_len * self.chann_num
        self.traj_len = get_length(self.traj)

        self.__check_params()
        self.__set_cable()
        self.__check_cable()


    def __check_params(self) -> None:
        """ Checks the input parameters.
        """

        # check the input trajectory
        if not isinstance(self.traj, np.ndarray) or \
                self.traj.shape[1] != 3 or self.traj.shape[0] < 2:
            raise ValueError("trajectory must be a 3D array with shape "
                             "(npts, 3), where npts >= 2")

        # check existence of duplicate coordinates
        if len(set(tuple(coord) for coord in self.traj)) != self.traj.shape[0]:
            raise ValueError("trajectory contains duplicate coordinates!")

        # check the input channel length and gauge length
        if self.chann_len < 0.0 or self.chann_len > self.traj_len:
            raise ValueError(
                f"Channel length must be between 0 and the trajectory length "
                f"({self.traj_len})")

        if self.gauge_len < 0.0 or self.gauge_len > self.traj_len:
            raise ValueError(
                f"Gauge length must be between 0 and the trajectory length "
                f"({self.traj_len})")

        # check the channel range, beiginning and ending
        if (self.chann_beg < self.gauge_len/2.0 or
                self.chann_beg > self.traj_len - self.gauge_len/2.0):
            raise ValueError(
                f"First channel length must be between {self.gauge_len/2.0} "
                f"and {self.traj_len - self.gauge_len/2.0} m, it is {self.chann_beg} m")

        if (self.chann_end < self.gauge_len/2.0 or
                self.chann_end > self.traj_len - self.gauge_len/2.0):
                
            print(
                f"Last channel length must be between {self.gauge_len/2.0} and "
                f"{self.traj_len - self.gauge_len/2.0} m, it is {self.chann_end} m")

            # change the channel number to make sure the last channel ends at the end of the cable
            self.chann_num = int((self.traj_len - self.chann_beg) / self.chann_len)
            self.chann_end = self.chann_beg + self.chann_len * self.chann_num

            print(
                f"Change the channel number to {self.chann_num} to make sure the "
                f"last channel ends at the end of the cable")


    def __set_cable(self) -> None:
        """ Set up the cable for computation.
        """
        
        # set the beginning of the trajectory
        traj_neg = select_along_traj(self.traj, self.chann_beg - self.gauge_len/2.0)
        traj_pos = select_along_traj(self.traj, self.chann_beg + self.gauge_len/2.0)
        traj_cha = select_along_traj(self.traj, self.chann_beg)

        # interpolate the trajector and select the desired channels
        traj_neg = interparc(traj_neg, self.chann_len)[:self.chann_num, :]
        traj_pos = interparc(traj_pos, self.chann_len)[:self.chann_num, :]
        traj_cha = interparc(traj_cha, self.chann_len)[:self.chann_num, :]

        # compute the tangent value of the points on the trajectory
        tan_neg, _, _, _, _ = frenet_serret(traj_neg)
        tan_pos, _, _, _, _ = frenet_serret(traj_pos)
        tan_cha, _, _, _, _ = frenet_serret(traj_cha)

        # collect the points on the trajectory and associated tangent vectors
        rec_locs = []
        tangent = []
        for ic in range(self.chann_num):
            rec_locs.append(traj_neg[ic])
            rec_locs.append(traj_pos[ic])
            tangent.append(tan_neg[ic])
            tangent.append(tan_pos[ic])

        # convert the list to array and round to four decimal places
        self.rec_locs = np.array(rec_locs)
        self.tangent = np.array(tangent)
        self.traj_cha = np.array(traj_cha)
        self.tan_cha = tan_cha

        # exclude close points and set lookup table for later use
        self.rec_locs_unique = remove_duplicates(self.rec_locs, self.threshold)

        # create a look-up table for the cable
        self.lookup_table = build_lookup_table(self.rec_locs, self.rec_locs_unique)

        # check the cable is 2D or 3D
        if np.all(np.isclose(self.traj[:, 1], 0.0)):
            self.ndim = 2
        else:
            self.ndim = 3


    def __check_cable(self) -> None:
        """ Check computed locations for each channel is correct, i.e., the
            distance between two neighboring points is equal to the channel
            length in numerical sense.

            Also check the gauge length is correct, i.e., the distance between
            the negative and positive channels is equal to the gauge length in
            numerical sense.

        Notes:
            There are may be some relative errors in specific cases, such as 
            at the vicinity of the bended area, as shown below. The computed 
            distance between two adjacent points is the Euclidean distance, 
            which may be slightly different from the arc length along the 
            trajectory. This will not affect the accuracy of the forward
            modeling. It is just because I compute the distance between two
            adjacent points using the Euclidean distance, instead of the arc
            length along the trajectory, to quantify the error.
            
                        |
                        |
                        *
                        |
                -----*--
        """

        # compute the error of the channel length
        dist_cha = np.linalg.norm(self.traj_cha[1:] - self.traj_cha[:-1], axis=1)
        dist_cha_err = np.max(abs(dist_cha-self.chann_len))

        # compute the error of the gauge length
        dis_gauge = np.zeros(self.chann_num)
        rec_coords = self.get_rec_loc_unique()
        for ic in range(self.chann_num):
            coord_neg = rec_coords[self.lookup_table[2 * ic]]
            coord_pos = rec_coords[self.lookup_table[2 * ic + 1]]
            coord_cha = self.traj_cha[ic]

            dis_gauge[ic] = np.linalg.norm(coord_cha - coord_neg) + \
                            np.linalg.norm(coord_pos - coord_cha)
            
        dis_gauge_err = np.max(abs(dis_gauge - self.gauge_len))

        self.dist_cha_err = dist_cha_err
        self.dis_gauge_err = dis_gauge_err
        
        # if dist_cha_err > 0.01 * self.chann_len:
        #     raise ValueError(
        #         f"Channel length error ({dist_cha_err}) exceeds 1% of the "
        #         f"channel length ({self.chann_len})")

        # if dis_gauge_err > 0.01 * self.gauge_len:
        #     raise ValueError(
        #         f"Gauge length error ({dis_gauge_err}) exceeds 1% of the gauge "
        #         f"length ({self.gauge_len})")
        

    def __repr__(self) -> str:
        """ Representation of the cable.
        """

        info = "DAS Cable Information: \n"
        info += f"  Cable total length: {self.traj_len:.2f} m \n"
        info += f"  Channel coverage: {self.chann_beg:.2f} - {self.chann_end:.2f} m \n"
        info += f"  Channel interval: {self.chann_len:.2f} m \n"
        info += f"  Channel number: {self.chann_num} \n"
        info += f"  Gauge length: {self.gauge_len:.2f} m \n"
        info += f"  Max channel length error: {self.dist_cha_err:.4f} m \n"
        info += f"  Max  gauge  length error: {self.dis_gauge_err:.4f} m \n"
        info += '\n'

        # # check dot product for a small time step
        # info += self.dot_product_test(nt=100, return_info=True)

        return info

    def get_chann_coords(self) -> np.ndarray:
        """ Returns the coordinates of the channels.
        """
        return self.traj_cha

    def get_tangent(self) -> np.ndarray:
        """ Returns the tangent vectors of the channels.
        """
        return self.tangent

    def get_rec_loc(self) -> np.ndarray:
        """ Returns the coordinates of the receivers (with duplicated).
        """
        return self.rec_locs

    def get_rec_loc_unique(self) -> np.ndarray:
        """ Returns the coordinates of the receivers (no duplicated).
        """
        return self.rec_locs_unique

    def forward(self, m: np.ndarray, 
                m_comp: str = 'vel', 
                d_comp: str = 'strain_rate', 
                dt: float = None) -> np.ndarray:
        """ Forward operator to generate DAS data:
                     d = Fm 
            where m is the three-component waveforms in the displacement form 
            (ux, uy, uz) or particle-velocity form (vx, vy, vz), and d is the 
            computed DAS response in the form of strain or strain rate.

        Parameters
        ----------
        m : ndarray
            The three-component waveforms in the displacement form (ux, uy, uz)
            or particle-velocity form (vx, vy, vz) with shape (3, nrec, nt).
        m_comp : str, optional (default: 'vel')
            The component of the input waveforms, can be either 'vel' (velocity)
            or 'disp' (displacement).
        d_comp : str, optional (default: 'strain_rate')
            The component of the output waveforms, can be either 'strain_rate'
            or 'strain'.
        dt : float, optional
            The time interval of the input waveforms, in seconds. Default is
            None.

        Returns
        -------
        d : ndarray
            The ndarray DAS data with shape (ncha, nt).

        Notes
        -----
            1. The optional parameter dt is required if integration for the case when 
            the input m is in the form of particle velocity and the output d is in
            the form of strain.
            2. For the 2-D case, the y coordinates of the input m must be all zeros.
            That is, m[1, :, :] = 0.0.
        """

        # check the input waveforms: m (3, nrec, nt)
        if (not isinstance(m, np.ndarray) or m.shape[0] != 3 or
                m.shape[1] != self.rec_locs_unique.shape[0]):
            raise ValueError("m must be a 3D array with shape "
                             "(3, rec_locs_unique, nt)")

        # check the component of the input waveforms and the output waveforms
        if m_comp not in ['vel', 'disp']:
            raise ValueError("m must be in the form of vel (velocity) "
                             "or disp (displacement)")
        if d_comp not in ['strain_rate', 'strain']:
            raise ValueError("d must be in the form of strain_rate "
                             "or strain")
        if  m_comp == 'vel' and d_comp == 'strain' and dt is None:
            raise ValueError("dt must be specified if m is in the form of "
                             "vel (velocity) and d is in the form of strain")

        if m_comp == 'disp' and d_comp == 'strain_rate' and dt is None:
            raise ValueError("dt must be specified if m is in the form of "
                             "disp (displacement) and d is in the form of strain_rate")
    

        # perform the forward modeling
        d = np.zeros((self.chann_num, m.shape[2]), dtype=m.dtype)

        for ic in range(self.chann_num):

            # obtain the channel number of the positive and negative channels
            channl_neg = 2 * ic
            channl_pos = 2 * ic + 1

            channl_neg_lookup = self.lookup_table[2 * ic]
            channl_pos_lookup = self.lookup_table[2 * ic + 1]

            # pull the model to the data space
            d[ic, :] = (np.dot(self.tangent[channl_pos], m[:, channl_pos_lookup, :]) -
                        np.dot(self.tangent[channl_neg], m[:, channl_neg_lookup, :]))

        # average the data over the gauge length
        d /= self.gauge_len

        # no other operator is needed
        if (m_comp == 'vel' and d_comp == 'strain_rate') or \
           (m_comp == 'disp' and d_comp == 'strain'):
            pass
        
        # integration operator is needed to convert strain rate to strain
        elif m_comp == 'vel' and d_comp == 'strain':
            d = np.cumsum(d, axis=-1) * dt

        # differentiation operator is needed to convert strain to strain rate
        elif m_comp == 'disp' and d_comp == 'strain_rate':
            d = np.diff(d, axis=-1, prepend=0.0) / dt

        return d
    

    def adjoint(self, d: np.ndarray, 
                m_comp: str = 'vel', 
                d_comp: str = 'strain_rate', 
                dt: float = None) -> np.ndarray: 
        """ Adjoint operator to generate receiver waveforms from DAS data:
                m = F'd
            where m is the three-component waveforms in the displacement form
            (ux, uy, uz) or particle-velocity form (vx, vy, vz), and d is the
            computed DAS response in the form of strain or strain rate. F' is
            the adjoint operator of F.

        Parameters
        ----------
        add : bool
            If True, add the computed receiver waveforms to the existing data.
        d : ndarray
            The ndarray DAS data with shape (ncha, nt).
        m_comp : str, optional
            The component of the input waveforms. Default is 'vel'.
        d_comp : str, optional
            The component of the output waveforms. Default is 'strain_rate'.
        dt : float, optional
            The time interval of the input waveforms, in seconds. Default is
            None.

        Returns
        -------
        m : ndarray
            The three-component waveforms in the displacement form (ux, uy, uz)
            or particle-velocity form (vx, vy, vz) with shape (3, nrec, nt).

        Notes
        -----
            1. The optional parameter dt is required if integration for the case when
            the output m is in the form of displacement and the input d is in
            the form of strain rate.
        """

        # check the input waveforms: d (ncha, nt)
        if (not isinstance(d, np.ndarray) or d.shape[0] != self.chann_num):
            raise ValueError("d must be a 2D array with shape "
                             "(chann_num, nt)")

        # check the component of the input waveforms and the output waveforms
        if m_comp not in ['vel', 'disp']:
            raise ValueError("m must be in the form of vel (velocity) "
                             "or disp (displacement)")
        if d_comp not in ['strain_rate', 'strain']:
            raise ValueError("d must be in the form of strain_rate "
                             "or strain")

        # perform the adjoint modeling
        m = np.zeros((3, self.rec_locs_unique.shape[0], d.shape[1]), dtype=d.dtype)

        #TODO: This part can be accelerated by numba or vectorized operations
        for ic in range(self.chann_num):

            # obtain the channel number of the positive and negative channels
            channl_neg = 2 * ic
            channl_pos = 2 * ic + 1

            channl_neg_lookup = self.lookup_table[2 * ic]
            channl_pos_lookup = self.lookup_table[2 * ic + 1]

            # push the data to the model space
            m[0, channl_pos_lookup, :] += d[ic, :] * self.tangent[channl_pos, 0]
            m[1, channl_pos_lookup, :] += d[ic, :] * self.tangent[channl_pos, 1]
            m[2, channl_pos_lookup, :] += d[ic, :] * self.tangent[channl_pos, 2]
            m[0, channl_neg_lookup, :] -= d[ic, :] * self.tangent[channl_neg, 0]
            m[1, channl_neg_lookup, :] -= d[ic, :] * self.tangent[channl_neg, 1]
            m[2, channl_neg_lookup, :] -= d[ic, :] * self.tangent[channl_neg, 2]

        # average the data over the gauge length
        m /= self.gauge_len

        # no other operator is needed
        if (m_comp == 'vel' and d_comp == 'strain_rate') or \
           (m_comp == 'disp' and d_comp == 'strain'):
            pass
        
        # adjoint of the integration operator (anti-causal integration)
        elif m_comp == 'vel' and d_comp == 'strain':
            m = np.flip(m, axis=-1)
            m = np.cumsum(m, axis=-1) * dt
            m = np.flip(m, axis=-1)

        # adjoint of the differentiation operator (negative derivative)
        elif m_comp == 'disp' and d_comp == 'strain_rate':
            m = -np.diff(m, axis=-1, append=0.0) / dt

        return m


    def dot_product_test(self, nt: int = 100, dt=0.001) -> None:
        """ Dot product of the forward and adjoint operators.
            d  = F m
            m' = F' d'
            
            dot(d', Fm) = dot(F'd', m), namely
            dot(d',  d) = dot(  m', m)

        Parameters
        ----------
        nt : int, optional
            The number of time samples. Default is 100.
        dt: float, optional
            The time interval of the input waveforms, in seconds. Default is
            0.001.
        Raises
        ------
        RuntimeError
            If the relative error is larger than 1e-6.
        """


        for m_comp in ['vel', 'disp']:
            for d_comp in ['strain_rate', 'strain']:

                # set data and model with random numbers
                rec_num = self.rec_locs_unique.shape[0]
                m = np.random.rand(3, rec_num, nt)
                d_titled = np.random.rand(self.chann_num, nt)

                # run forward operator
                d = self.forward(m, m_comp=m_comp, d_comp=d_comp, dt=dt)

                # run adjoint operator
                m_titled = self.adjoint(d_titled, m_comp=m_comp, d_comp=d_comp, dt=dt)

                # perform dot product
                lhs = np.dot(d_titled.flatten(), d.flatten())
                rhs = np.dot(m_titled.flatten(), m.flatten())

                # print results
                info = "********************************************** \n"
                info += "Dot-product test: \n"
                info += f"  Input waveform comp: {m_comp}, DAS comp: {d_comp} \n"
                info += "  lhs = %.8e \n" % (lhs)
                info += "  rhs = %.8e \n" % (rhs)
                info += "  absolute error = %.8e \n" % (abs(lhs-rhs))
                info += "  relative error = %.8e \n" % (abs(lhs-rhs)/abs(lhs))
                if (abs(lhs-rhs)/abs(lhs) > 1e-6):
                    raise RuntimeError("Dot-product test failed!")

                print(info)

    def plot_traj(self, save_path: str = None) -> None:
        """ Plot the trajectory of the control points.

        Parameters
        ----------
        save_path : str, optional
            The path to save the figure. If not specified, the figure will
            be shown directly.
        """

        # Two dimensions (plane curve)
        if self.ndim == 2:
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(1, 1, 1)
            plt.plot(self.traj[:, 0], self.traj[:, 2], '-o', 
                     color='k', linewidth=0.8, markersize=6.0)
            
            # set the range
            xextend = 200.0
            zextend = 200.0
            xspan = np.max(self.traj[:,0]) -  np.min(self.traj[:,0]) + xextend
            zspan = np.max(self.traj[:,2]) -  np.min(self.traj[:,2]) + zextend

            if xspan < 0.3 * zspan:
                xextend = zspan * 0.3
            elif zspan < 0.3 * xspan:
                zextend = xspan * 0.3

            plt.xlim(np.min(self.traj[:,0]) - xextend, np.max(self.traj[:,0]) + xextend,)
            plt.ylim(np.min(self.traj[:,2]) - zextend, np.max(self.traj[:,2]) + zextend,)

            ax.grid(True)
            ax.set_aspect('equal')
            ax.set_xlabel('Distance (m)', fontsize=12)
            ax.set_ylabel('Depth (m)', fontsize=12)
            ax.set_title('Trajectory of Control Points', fontsize=12)
            ax.invert_yaxis()

        # Three dimensions (space curve)
        elif self.ndim == 3:
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            plt.plot(self.traj[:, 0], self.traj[:, 1], self.traj[:, 2], '-o', 
                     color='k', linewidth=0.8, markersize=3.0)
            
            # set the range
            ax.set_xlim(np.min(self.traj[:,0]) - 100.0, np.max(self.traj[:,0]) + 100.0,)
            ax.set_ylim(np.min(self.traj[:,1]) - 100.0, np.max(self.traj[:,1]) + 100.0,)
            ax.set_zlim(np.min(self.traj[:,2]) - 100.0, np.max(self.traj[:,2]) + 100.0,)
            
            ax.grid(True)
            ax.set_box_aspect([1, 1, 1]) # must be equal, otherwise the direction is messed up
            ax.set_xlabel('Distance-x (m)', fontsize=12, color='k')
            ax.set_ylabel('Distance-y (m)', fontsize=12, color='k')
            ax.set_zlabel('Depth (m)', fontsize=12, color='k')
            ax.set_title('Trajectory of Control Points', fontsize=12)
            ax.invert_zaxis()

        
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
            

    def plot_channel(self, show_tangent: Optional[bool] = False,
                     save_path: Optional[str] = None) -> None:
        """ Plot the trajectory of the channels.
        
        Parameters
        ----------
        show_tangent : bool, optional
            If True, plot the tangent vectors of the interpolated points.

        save_path : str, optional
            The path to save the figure. If not specified, the figure will
            be shown directly.
        """

        # Two dimensions (plane curve)
        if self.ndim == 2:
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(1, 1, 1)
            plt.plot(self.traj_cha[:, 0], self.traj_cha[:, 2], 
                     '-o', color='k', linewidth=0.8, markersize=6.0)
            
            # set the range
            xextend = 200.0
            zextend = 200.0
            xspan = np.max(self.traj[:,0]) -  np.min(self.traj[:,0]) + xextend
            zspan = np.max(self.traj[:,2]) -  np.min(self.traj[:,2]) + zextend

            if xspan < 0.3 * zspan:
                xextend = zspan * 0.3
            elif zspan < 0.3 * xspan:
                zextend = xspan * 0.3

            plt.xlim(np.min(self.traj[:,0]) - xextend, np.max(self.traj[:,0]) + xextend,)
            plt.ylim(np.min(self.traj[:,2]) - zextend, np.max(self.traj[:,2]) + zextend,)


            # plot the tangent vectors of the interpolated points
            # Note: I added the minus sign to flip the z axis, so that the depth is positive downward
            if show_tangent: 
                ax.quiver(self.traj_cha[:, 0], self.traj_cha[:, 2], 
                          self.tan_cha[:, 0], -self.tan_cha[:, 2], 
                          color='r', scale=20.0, width=0.003)

            ax.grid(True)
            ax.set_aspect('equal')
            ax.set_xlabel('Distance (m)', fontsize=12)
            ax.set_ylabel('Depth (m)', fontsize=12)
            ax.set_title('Trajectory of Channels', fontsize=12)
            ax.invert_yaxis()

        # Three dimensions (space curve)
        elif self.ndim == 3:
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(1, 1, 1, projection='3d')

            plt.plot(self.traj_cha[:, 0], self.traj_cha[:, 1], self.traj_cha[:, 2], 
                     '-o', color='k', linewidth=0.8, markersize=3.0)
            # set the range
            ax.set_xlim(np.min(self.traj[:,0]) - 100.0, np.max(self.traj[:,0]) + 100.0,)
            ax.set_ylim(np.min(self.traj[:,1]) - 100.0, np.max(self.traj[:,1]) + 100.0,)
            ax.set_zlim(np.min(self.traj[:,2]) - 100.0, np.max(self.traj[:,2]) + 100.0,)
            
            # plot the tangent vectors of the interpolated points
            if show_tangent: 
                ax.quiver(self.traj_cha[:, 0], self.traj_cha[:, 1],
                          self.traj_cha[:, 2], self.tan_cha[:, 0],
                          self.tan_cha[:, 1], self.tan_cha[:, 2],
                          color='r', length=30.0, normalize=True)

            ax.grid(True)
            ax.set_box_aspect([1, 1, 1]) # must be equal, otherwise the direction is messed up
            ax.set_xlabel('Distance-x (m)', fontsize=12, color='k')
            ax.set_ylabel('Distance-y (m)', fontsize=12, color='k')
            ax.set_zlabel('Depth (m)', fontsize=12, color='k')
            ax.set_title('Trajectory of Channels', fontsize=12)
            ax.invert_zaxis()

        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_sensitivity(self, save_path: str = None) -> None:
        """ Plot the sensitivity of the channels.

        Parameters
        ----------
        save_path : str, optional
            The path to save the figure. If not specified, the figure will
            be shown directly.
        """

        # Two dimensions (plane curve)
        if self.ndim == 2:

            fig = plt.figure(figsize=(16, 8))

            data = [self.tan_cha[:, 0], self.tan_cha[:, 2]]
            titles = ['Sensitivity to Vx', 'Sensitivity to Vz']
            for i, d in enumerate(data):
                ax = fig.add_subplot(1, len(data), i+1)
                cs = ax.scatter(self.traj_cha[:, 0], self.traj_cha[:, 2],
                                c=d, vmin=-1.0, vmax=1.0, cmap='jet', s=10)
                # add colorbar to the bottom of the figure, tight
                plt.colorbar(cs, orientation='horizontal',
                             fraction=0.04, pad=0.08)
                # set the range
                xextend = 200.0
                zextend = 200.0
                xspan = np.max(self.traj[:,0]) -  np.min(self.traj[:,0]) + xextend
                zspan = np.max(self.traj[:,2]) -  np.min(self.traj[:,2]) + zextend

                if xspan < 0.3 * zspan:
                    xextend = zspan * 0.3
                elif zspan < 0.3 * xspan:
                    zextend = xspan * 0.3

                plt.xlim(np.min(self.traj[:,0]) - xextend, np.max(self.traj[:,0]) + xextend,)
                plt.ylim(np.min(self.traj[:,2]) - zextend, np.max(self.traj[:,2]) + zextend,)

                ax.grid(True)
                ax.set_aspect('equal')
                ax.set_xlabel('Distance (m)', fontsize=12)
                ax.set_ylabel('Depth (m)', fontsize=12)
                ax.set_title(titles[i], fontsize=16)
                ax.invert_yaxis()

        # Three dimensions (space curve)
        elif self.ndim == 3:

            fig = plt.figure(figsize=(16, 8))

            data = [self.tan_cha[:, 0], self.tan_cha[:, 2]]
            titles = ['Sensitivity to Vx', 'Sensitivity to Vz']
            for i, d in enumerate(data):
                ax = fig.add_subplot(1, len(data), i+1, projection='3d')
                cs = ax.scatter(self.traj_cha[:, 0], self.traj_cha[:, 1], self.traj_cha[:, 2],
                                c=d, vmin=-1.0, vmax=1.0, cmap='jet', s=10)
                # add colorbar to the bottom of the figure, tight
                plt.colorbar(cs, orientation='horizontal',
                             fraction=0.04, pad=0.08)
                
                ax.set_xlim(np.min(self.traj[:,0]) - 100.0, np.max(self.traj[:,0]) + 100.0,)
                ax.set_ylim(np.min(self.traj[:,1]) - 100.0, np.max(self.traj[:,1]) + 100.0,)
                ax.set_zlim(np.min(self.traj[:,2]) - 100.0, np.max(self.traj[:,2]) + 100.0,)
                
                ax.grid(True)
                ax.set_box_aspect([1, 1, 1]) # must be equal, otherwise the direction is messed up
                ax.set_xlabel('Distance (m)', fontsize=12)
                ax.set_ylabel('Distance (m)', fontsize=12)
                ax.set_zlabel('Depth (m)', fontsize=12)
                ax.set_title(titles[i], fontsize=16)
                ax.invert_zaxis()

        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def get_length(traj: np.ndarray) -> float:
    """ Returns the length of the cable
    
    Parameters
    ----------
    traj : np.ndarray
        The trajectory of the cable. Shape (npts, 3)
    
    Returns
    -------
    float
        The length of the cable
    """

    if traj.shape[1] != 3 or traj.shape[0] < 2:
        raise ValueError("trajectory must be a 3D array with shape (npts, 3), "
                         "where npts >= 2")

    return np.sum(np.sqrt(np.sum(np.diff(traj, axis=0)**2, axis=1)))


def select_along_traj(traj: np.ndarray, arc_len: float) -> np.ndarray:
    """Select the point along the trajectory of the das cable

    Parameters
    ----------
    traj : np.ndarray
        The trajectory of the cable. Shape (npts, 3)
    arc_len : float
        The arclength of the point of interest
    
    Returns
    -------
    np.ndarray
        The coordinates of the point of interest. Shape (3,)
    """

    # convert to np.float32
    traj = np.array(traj, dtype=np.float32)

    # check the input trajectory
    if traj.shape[1] != 3 or traj.shape[0] < 2:
        raise ValueError("trajectory must be a 3D array with shape (npts, 3), "
                         "where npts >= 2")

    # calculate the chord length of the das cable
    chord_lens = np.sqrt(np.sum(np.diff(traj, axis=0) ** 2, axis=1))
    chord_len_total = np.sum(chord_lens)

    # check the input arclength
    if arc_len < 0 or arc_len > chord_len_total:
        raise ValueError(
            f"arclength must be within the range of [0, {chord_len_total}]")

    # find the index of the point of interest
    accum_len = 0.0
    for ip in range(traj.shape[0]):
        accum_len += chord_lens[ip]
        if accum_len >= arc_len:
            break

    # compute the desired point along the das cable by linear interpolation
    r = (accum_len - arc_len) / chord_lens[ip]
    l = 1.0 - r

    new_point = r * traj[ip, :] + l * traj[ip+1, :]

    # insert the new point into the trajectory
    new_point = np.array(new_point, dtype=np.float32)
    new_traj = np.insert(traj, ip+1, new_point, axis=0)
    new_traj = new_traj[ip+1:, :]

    return new_traj



def interparc(p: np.ndarray, interval: float, method: str = 'linear', 
              verbose: bool = False) -> np.ndarray:
    """ Interpolate points along a curve in 2 or more dimensions.

    Parameters
    ----------
    p : np.array
        The coordinates of the points defining the curve to be interpolated. 
        Must be of shape (npts, 3)
    interval : float
        The desired gauge length of the interpolated points
    method : str, optional
        The interpolation method to use. Either 'linear' or 'spline'.
        The default is 'linear'.
    verbose : bool, optional
        Print out the estimated number of points and the averaged error.

    Returns
    -------
    np.array
        The interpolated points. Shape (npts, 3)
    """

    # check method
    valid_methods = ['linear', 'spline']

    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}")
    elif method == 'spline':
        raise NotImplementedError(
            "Spline interpolation is not yet implemented")

    # check interval
    if interval <= 0:
        raise ValueError("interval must be positive")

    # check p
    npts = p.shape[0]
    ndim = p.shape[1]

    if ndim not in [2, 3]:
        raise ValueError("p must be 2D or 3D")

    if npts < 2:
        raise ValueError("p must have at least 2 points")

    # Calculate the approximated npts
    chordlen_all = np.sqrt(np.sum(np.diff(p, axis=0) ** 2, axis=1))
    chordlen = np.sum(chordlen_all)

    if chordlen <= interval:
        raise ValueError(f"Gauge length {interval} must be smaller than the "
                         "length of the curve {chordlen}")

    nt = int(np.floor(chordlen / interval))

    # adjust the last point to make sure the arc length is dividable by interval

    # length of interval * (nt-1)
    chordlen_desired = interval * nt

    # calculate the distance needed to be adjusted to the last point
    d = chordlen_desired - chordlen + chordlen_all[-1]

    # calculate the unit vector of the last segment
    u = (p[-1] - p[-2]) / chordlen_all[-1]

    # add the distance to the last point, assuming the last segment is straight
    p[-1] = p[-2] + d * u

    # calculate the new chord length
    chordlen_all = np.sqrt(np.sum(np.diff(p, axis=0) ** 2, axis=1))
    chordlen = np.sum(chordlen_all)

    nt = int(np.floor(chordlen / interval)) + 1

    # interpolate the curve with the desired number of points
    pt = interpolate_curve(nt, p, npts, ndim)

    if verbose:
        # calculate the error with the desired gauge length
        error = np.mean(
            abs(np.sqrt(np.sum(np.diff(pt, axis=0) ** 2, axis=1)) - interval))
        print(f"Desired interval: {interval}")
        print(f"Number of points: {nt}, with averaged error: {error:.6f}")

    # dudt = (p[tbins] - p[tbins-1]) / chordlen[tbins-1][:, np.newaxis]

    return pt


def interpolate_curve(nt: int, p: np.ndarray, npts: int, ndim: int) -> np.ndarray:
    """ Interpolate the curve with the desired number of points

    Parameters
    ----------
    nt : int
        The desired number of points
    p : np.array
        The coordinates of the points defining the curve to be interpolated.
        Must be of shape (npts, 3)
    npts : int
        The number of points in the original curve
    ndim : int
        The number of dimensions of the curve

    Returns
    -------
    np.array
        The interpolated points. Shape (npts, 3)
    """

    t = np.linspace(0, 1, nt)

    pt = np.full((nt, ndim), np.nan)
    chordlen = np.sqrt(np.sum(np.diff(p, axis=0)**2, axis=1))
    chordlen /= np.sum(chordlen)
    cumarc = np.hstack((0, np.cumsum(chordlen)))

    tbins = np.digitize(t, cumarc)
    tbins[(tbins <= 0) | (t <= 0)] = 1
    tbins[(tbins >= npts) | (t >= 1)] = npts - 1
    s = (t - cumarc[tbins-1]) / chordlen[tbins-1]
    pt = p[tbins-1] + (p[tbins] - p[tbins-1]) * s[:, np.newaxis]

    return pt


def frenet_serret(traj: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ Calculate the Frenet-Serret Space Curve Invariants
        _    r'
        T = ----  (Tangent)
            |r'|

        _    T'
        N = ----  (Normal)
            |T'|
        _   _   _
        B = T x N (Binormal)

        k = |T'|  (Curvature)

        t = dot(-B',N) (Torsion)

    Parameters
    ----------
    traj : np.array
        The trajectory of the curve. Must be of shape (npts, 3)
    
    Returns
    -------
    T : np.array
        The tangent vector. Shape (npts, 3)
    N : np.array
        The normal vector. Shape (npts, 3)
    B : np.array    
        The binormal vector. Shape (npts, 3)
    k : np.array
        The curvature. Shape (npts,)
    t : np.array
        The torsion. Shape (npts,)

    Notes
    -----
    The code below is benchmarked with MATLAB code: 
        Daniel Claxton (2023). Frenet (https://www.mathworks.com/matlabcentral/
        fileexchange/11169-frenet), MATLAB Central File Exchange. Retrieved 
        August 14, 2023.

    The Tanget is verified to be accurate, while the Normal and Binormal both 
    have some descrepancies, only at a few points. I check the reason and it is
    because of division of two very small numbers. I think it is fine to
    ignore these descrepancies.
    """

    # get the x, y, z coordinates of the trajectory
    x = traj[:, 0]
    y = traj[:, 1]
    z = traj[:, 2]

    # Check input
    if x.shape != y.shape != z.shape:
        raise ValueError("x, y, z must have the same shape")

    # Convert to column vectors
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()

    # Speed of curve
    dx = np.gradient(x)
    dy = np.gradient(y)
    dz = np.gradient(z)
    dr = np.vstack((dx, dy, dz)).T

    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    ddz = np.gradient(dz)
    ddr = np.vstack((ddx, ddy, ddz)).T

    # Tangent
    T = dr / mag(dr, 3)

    # Derivative of tangent
    dTx = np.gradient(T[:, 0])
    dTy = np.gradient(T[:, 1])
    dTz = np.gradient(T[:, 2])
    dT = np.vstack((dTx, dTy, dTz)).T

    # Normal: the division of two very small numbers may have some descrepancies
    N = dT / mag(dT, 3)

    # Binormal
    B = np.cross(T, N)

    # Curvature
    k = mag(np.cross(dr, ddr), 1) / ((mag(dr, 1))**3)

    # Torsion
    t = -np.einsum('ij,ij->i', B, N)

    # Return the normal
    return T, N, B, k, t


def mag(T: np.ndarray, n: int) -> np.ndarray:
    """ Magnitude of a vector (Nx3)

    Parameters
    ----------
    T : np.array
        The vector. Shape (npts, 3)
    n : int
        The dimension of the vector
    
    Returns
    -------
    np.array
        The magnitude of the vector. Shape (npts,)
    """

    M = np.linalg.norm(T, axis=1)
    d = np.where(M == 0)[0]
    M[d] = np.finfo(float).eps * np.ones_like(d)
    M = M[:, np.newaxis]
    M = np.tile(M, (1, n))

    return M


def remove_duplicates(coords: np.ndarray, threshold: float) -> np.ndarray:
    """ Remove duplicate coordinates from the coords array

    Parameters
    ----------        
        coords (np.ndarray): 2D array of coordinates
        threshold (float): threshold distance to consider two points as the same

    Returns
    -------
        unique_coords (np.ndarray): 2D array of unique coordinates
    """

    unique_coords = []
    for coord in coords:
        is_duplicate = False
        for unique_coord in unique_coords:
            if np.linalg.norm(coord - unique_coord) <= threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_coords.append(coord)

    return np.array(unique_coords)


def build_lookup_table(coords: np.ndarray, unique_coords: np.ndarray) -> np.ndarray:
    """ Build a lookup table to map the coords to the unique_coords

    Parameters
    ----------
        coords (np.ndarray): 2D array of coordinates
        unique_coords (np.ndarray): 2D array of unique coordinates
    
    Returns
    -------
        lookup_table (np.ndarray): 1D array of indices
    """

    lookup_table = []
    for coord in coords:
        nearest_unique_idx = np.argmin(
            np.linalg.norm(unique_coords - coord, axis=1))
        lookup_table.append(nearest_unique_idx)

    return lookup_table



def smart_cable(beg, end, depth, well_interval, well_depth, well_width=0.1):
    """
    Generate a trajectory for a smart cable used in seismic modeling and inversion

    Parameters:
        beg (float): The starting position of the cable.
        end (float): The ending position of the cable.
        depth (float): The depth of the cable.
        well_interval (float): The interval between wells along the cable.
        well_depth (float): The depth of the wells.
        well_width (float, optional): The width of the wells. Defaults to 0.1.

    Returns:
        numpy.ndarray: An array containing the trajectory of the smart cable.

    Raises:
        AssertionError: If `beg` is greater than or equal to `end`.
        AssertionError: If `well_interval` is greater than or equal to the difference between `end` and `beg`.

    """

    assert beg < end, "must be increasing"
    assert well_interval < end - beg, "cable is not long enough even for one well"
    x = beg
    y = 0.0  # 2-D case

    traj = []

    # cable head
    traj.append([x, y, depth])

    while True:

        x = x + well_interval
        if x + well_width >= end:
            break

        # downgoing section of the well
        traj.append([x, y, depth])
        traj.append([x, y, depth + well_depth])

        # upgoing section of the well
        x += well_width
        traj.append([x, y, depth + well_depth])
        traj.append([x, y, depth])

    # cable end
    traj.append([end, y, depth])

    return np.array(traj)

