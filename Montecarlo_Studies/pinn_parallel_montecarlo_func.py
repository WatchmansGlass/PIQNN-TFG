def paralel_montecarlo_pinn_schrodinger(j,max_epochs=2000, n_neurons_1=4, n_neurons_2=7,  seed=128):

    import os

    #Usamos el Keras 2.x, ya que es el único compatible con qml.qnn.KerasLayer

    os.environ["TF_USE_LEGACY_KERAS"] = "1"

    import numpy as np
    import tensorflow as tf
    import matplotlib.pyplot as plt
    import time
    import scipy
    from scipy.linalg import eigh_tridiagonal


    #Evitamos warnings sobre el cast complex a float

    tf.get_logger().setLevel('ERROR')

    #Forzamos a Tensorflow a trabajar con "float64" (desactivado)

    #tf.keras.backend.set_floatx('float64')


    num_domain = 64
    l = 1
    minval = -6*l
    maxval = 6*l

    def generate_uniform_data(num_domain, minval, maxval):

      data_init = tf.random_uniform_initializer(minval=minval, maxval=maxval, seed=seed)
      return tf.Variable(data_init(shape=[num_domain, 1]), dtype=tf.float32)

    def generate_uniform_simetric_data(num_domain, minval, maxval):

      data_init = tf.random_uniform_initializer(minval=minval, maxval=(maxval+minval)/2, seed=seed)
      x_left = data_init(shape=[num_domain//2, 1])
      x_left = tf.sort(x_left, axis=0)
      x_right = -1*x_left[::-1]
      x_left = tf.Variable(x_left, dtype=tf.float32)
      x_right = tf.Variable(x_right, dtype=tf.float32)
      x = tf.concat([x_left, x_right], 0)
      return x


    hbar = 1
    m = 1
    v0 = 50

    def V(x):

      Xs = x
      Vnp = (1 - np.heaviside(x+l, 0.5) + np.heaviside(x-l, 0.5))*v0
      V = tf.convert_to_tensor(Vnp, dtype=tf.float32)

      return V

    def g(x):

      Xs = x
      gnp = (1 - np.exp(-abs(Xs - minval)))*(1 - np.exp(-abs(Xs - maxval)))
      g = tf.convert_to_tensor(gnp, dtype=tf.float32)

      return g

    x = generate_uniform_simetric_data(num_domain//2, minval, maxval)
    conc_x = generate_uniform_simetric_data(num_domain//2, -l, l)
    x = tf.concat([x, conc_x], axis=0)
    x = tf.sort(x, axis=0)
    x = tf.Variable(x, dtype=tf.float32)
    #print(x)
    V_pot = V(x)




    def pde(tape1, tape2, x, y, w, b, j=0):
      dy_x = tape1.gradient(y, x)[:, j : j + 1]
      dy_xx = tape2.gradient(dy_x, x)[:, j : j + 1]
      return -(hbar**2)/(2*m) * dy_xx + (V_pot - (w+b))*y

    def bc_i(x):
      return 0

    def bc_f(x):
      return 0

    x_bc_i = tf.constant(minval, shape = [1, 1], dtype=tf.float32)
    x_bc_f = tf.constant(maxval, shape = [1, 1], dtype=tf.float32)
    x_0 = tf.constant(0, shape = [1, 1], dtype=tf.float32)


    ###

    n_inputs = 1
    n_outputs = 1
    activation = 'tanh'

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input((n_inputs,)))
    model.add(tf.keras.layers.Dense(units=8, activation=activation))
    model.add(tf.keras.layers.Dense(units=n_neurons_1, activation=activation))
    model.add(tf.keras.layers.Dense(units=n_neurons_2, activation=activation))
    model.add(tf.keras.layers.Dense(units=n_outputs))

    if j==0:

      model.summary()


    epochs = max_epochs
    learning_rate = 0.02

    flag = False
    previous_mse = 0
    diff_tolerance = 0.04
    tolerance = 0.4
    max_count = 5
    count = 0

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    error_arr = np.zeros(epochs//20 + 1)

    epoch_arr = np.linspace(0, epochs, epochs//20 + 1)

    E = 1.8570135450324359

    w, b = E, 0



    while not flag:

      starting_time = time.time()

      for i in range(epochs + 1):
        with tf.GradientTape() as tape_model:
          with tf.GradientTape() as tape_pde:
            with tf.GradientTape() as tape_pde2:


              y = model(x, training=True)
              #w, b = model.get_weights()[0][0][0], model.get_weights()[1][0]
              bc_i_error = model(x_bc_i, training=True)[0][0] - bc_i(x_bc_i)
              bc_f_error = model(x_bc_f, training=True)[0][0] - bc_f(x_bc_f)
              pos_error = tf.reduce_mean(tf.square(tf.minimum(y, 0.0)))

              gf = g(x)

              f = y*gf

              tape_pde.watch(x)
              tape_pde2.watch(x)

              domain_error = pde(tape_pde, tape_pde2, x, f, w, b)



          domain_mse = mse = tf.math.reduce_mean(tf.math.square(domain_error), axis=0)

          loss_norm = tf.math.square(tf.tensordot(f, f, 2) / E**2 - tf.convert_to_tensor(num_domain/(maxval - minval), dtype = tf.float32))
          #norm = np.sqrt(scipy.integrate.simpson(tf.tensordot(abs(f), tf.transpose(abs(f)), 1), x[:, 0]))
          #print(np.shape(f[:, 0]), np.shape(x))
          #print(norm, np.sqrt(scipy.integrate.simpson(abs(f[:, 0])**2, x[:, 0])))
          #loss_norm = tf.math.square(tf.reduce_sum(f[:, 0] - f[:, 0]/norm))
          #loss_norm = tf.math.square(tf.math.reduce_sum(f[:, 0]*f[:, 0] - 1))
          #loss_eigen = tf.convert_to_tensor(np.exp(abs(-(w+b) + E)), dtype = tf.float32) - 1
          loss_sim = tf.math.reduce_mean(tf.math.exp(tf.math.square(f[:len(f)//2 , 0] - f[len(f)//2:,0][::-1])) - 1)
          bc_i_mse =  tf.math.square(bc_i_error)
          bc_f_mse =  tf.math.square(bc_f_error)

          total_mse = domain_mse + 20*loss_norm + 20*bc_i_mse + 20*bc_f_mse + 20*loss_sim
          total_mse += (200*pos_error)**2
          #total_mse += loss_eigen


          #if i % 100 == 0:
            #print('Epoch: {}\t MSE Loss = {}'.format(i, total_mse.numpy()[0]))
            #print('Domain Loss {}'.format(domain_mse.numpy()[0]))
            #print('Boundary Loss {}'.format(20*bc_i_mse + 20*bc_f_mse))
            #print('Normalization Loss {}'.format(20*loss_norm))
            #print('Eigenvalue Loss {}'.format(loss_eigen))
            #print('Simetry Loss {}'.format(20*loss_sim))
            #print('Positive Error {}'.format((200*pos_error)**2))

          if i % 20 == 0:

            error_arr[i//20] = total_mse.numpy()[0]

          if i == 0:

            first_error = total_mse.numpy()[0]

            print('Montecarlo: {}\t Starting MSE Loss = {}'.format(j+1, first_error), flush=True)




        if abs(total_mse - previous_mse) < diff_tolerance and total_mse < tolerance:

          count += 1

          if count >= max_count:

            max_epoch = i
            last_error = total_mse.numpy()[0]


            print('Montecarlo: {}\t Final Epoch = {}\t Final MSE Loss = {}'.format(j+1, max_epoch, last_error), flush=True)

            flag = True
            break


        else:

          count = 0

        if i == epochs:

          max_epoch = i
          last_error = total_mse.numpy()[0]


          print('Montecarlo: {}\t Final Epoch = {}\t Final MSE Loss = {}'.format(j+1, max_epoch, last_error), flush=True)

          flag = True
          break

        previous_mse = total_mse

        model_update_gradients = tape_model.gradient(total_mse, model.trainable_variables)
        optimizer.apply_gradients(
        zip(model_update_gradients, model.trainable_variables)
        )

      ending_time = time.time()

      print(f"Execution time of NN number {j+1}: ", ending_time-starting_time, flush=True)

      x_test = np.linspace(minval, maxval, 601)
      #print(x_test[:, 0])
      gf_test = g(x_test)
      y_pred = model(x_test)
      f_pred = gf_test*y_pred[:,0]
      f_pred_norm = f_pred / np.sqrt(scipy.integrate.simpson(f_pred**2, x_test))
      V_plot = V(x_test)



      V_d = np.where(np.abs(x_test) <= l, 0.0, v0)

      dx = x_test[1] - x_test[0]

      main_diag = 2.0 / dx**2 + V_d[1:-1]
      off_diag = -1.0 / dx**2 * np.ones(601 - 3)

      eigenvalues, eigenvectors = eigh_tridiagonal(main_diag, off_diag)

      E0 = eigenvalues[0]
      psi0 = eigenvectors[:, 0]



      psi0 = psi0 / np.sqrt(scipy.integrate.simpson(abs(psi0)**2, x_test[1:-1]))

      psi_full = np.zeros(601)
      psi_full[1:-1] = psi0

      y_true = psi_full

      val_error = 0

      for i in range(len(f_pred)):

        val_error += (f_pred[i] - y_true[i])**2

      val_error = val_error/len(f_pred)
      val_error = val_error

      print('Test MSE Loss = {}'.format(val_error), flush=True)
      
      return error_arr, epoch_arr, [first_error, max_epoch, last_error, val_error]


##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################


def paralel_montecarlo_pinn_pendulum(j,max_epochs=10000, n_nodes=8, seed=128):
    
    import os
    
    #Usamos el Keras 2.x, ya que es el único compatible con qml.qnn.KerasLayer
    
    os.environ["TF_USE_LEGACY_KERAS"] = "1"
    
    import numpy as np
    import tensorflow as tf
    #import matplotlib.pyplot as plt
    import pennylane as qml
    import time
    

    #Evitamos warnings sobre el cast complex a float
    
    tf.get_logger().setLevel('ERROR')
    
    #Forzamos a Tensorflow a trabajar con "float64" (desactivado)
    
    #tf.keras.backend.set_floatx('float64')
        
    
    num_domain = 64
    minval = 0
    maxval = 2.5
    
    def generate_uniform_data(num_domain, minval, maxval):
      data_init = tf.random_uniform_initializer(minval=minval, maxval=maxval, seed=seed)
      return tf.Variable(data_init(shape=[num_domain, 1]), dtype=tf.float32)
    
    x = generate_uniform_data(num_domain, minval, maxval)
    #gamma = 1
    g = 9.8
    L = 1
    w = np.sqrt(g/L)
    T = 2*np.pi/w
    
    ###
    
    def pde(tape1, tape2, x, y, j=0):
      dy_x = tape1.gradient(y, x)[:, j : j + 1]
      dy_xx = tape2.gradient(dy_x, x)[:, j : j + 1]
      return dy_xx + g/L * tf.sin(y)
    
    def ic_0(x):
    
      return np.pi/4
    
    def ic_0_d(x):
    
      return 0
    
    
    x_0 = tf.Variable([[0]], shape=[1, 1], dtype=tf.float32)
    

    ###
    
    n_inputs = 1
    n_outputs = 1
    activation = 'tanh'
    
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input((n_inputs,)))
    model.add(tf.keras.layers.Dense(units=4, activation=activation))
    model.add(tf.keras.layers.Dense(units=4, activation=activation))
    model.add(tf.keras.layers.Dense(units=n_nodes, activation=activation))
    model.add(tf.keras.layers.Dense(units=n_outputs))
  
    epochs = max_epochs
    learning_rate = 0.02
    
    flag = False
    previous_mse = 0
    diff_tolerance = 0.0003
    tolerance = 0.003
    max_count = 5
    count = 0
  
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
  
    error_arr = np.zeros(epochs//20 + 1)
  
    epoch_arr = np.linspace(0, epochs, epochs//20 + 1)
  
    while not flag:
      
      starting_time = time.time()
      
      for i in range(epochs + 1):
          
          
        with tf.GradientTape() as tape_model:
            
          with tf.GradientTape() as tape_bc:

              y_0 = model(x_0, training=True)
              dy_0 = tape_bc.gradient(y_0, x_0)[0,0]
              ic2_error = dy_0 - ic_0_d(x_0)

              ic_error = abs(model(x_0, training=True)[0][0] - ic_0(x_0))
        
                      
          with tf.GradientTape() as tape_pde:
            with tf.GradientTape() as tape_pde2:
                y = model(x, training=True)

                domain_error = pde(tape_pde, tape_pde2, x, y)
  
          domain_mse = mse = tf.math.reduce_mean(tf.math.square(domain_error), axis=0)
          ic_mse =  tf.math.square(ic_error) / int(maxval/T)
          ic2_mse = tf.math.exp(1.+tf.math.square(ic2_error)) - tf.math.exp(1.)
          total_mse = domain_mse + 20*ic_mse + 20*ic2_mse
  
          #if i % 100 == 0:

            #print('Epoch: {}\t Total Loss = {}'.format(i, total_mse.numpy()[0]))
            #print('Domain Loss = {}'.format(domain_mse.numpy()[0]))
            #print('Initial and Periodical Condition Loss = {}'.format(ic_mse.numpy() + ic2_mse.numpy()))
            #print('Experimental Loss = {}'.format(exp_mse.numpy()[0]))
  
          if i % 20 == 0:
  
            error_arr[i//20] = total_mse.numpy()[0]
  
          if i == 0:
  
            first_error = total_mse.numpy()[0]
  
            print('Montecarlo: {}\t Starting MSE Loss = {}'.format(j+1, first_error), flush=True)
  
  
  
  
        if abs(total_mse - previous_mse) < diff_tolerance and total_mse < tolerance:
  
          count += 1
  
          if count >= max_count:
  
            max_epoch = i
            last_error = total_mse.numpy()[0]
  
  
            print('Montecarlo: {}\t Final Epoch = {}\t Final MSE Loss = {}'.format(j+1, max_epoch, last_error), flush=True)
  
            flag = True
            break
  
  
        else:
  
          count = 0
  
        if i == epochs:
  
          max_epoch = i
          last_error = total_mse.numpy()[0]
  

          print('Montecarlo: {}\t Final Epoch = {}\t Final MSE Loss = {}'.format(j+1, max_epoch, last_error), flush=True)
  
          flag = True
          break
  
        previous_mse = total_mse
  
        model_update_gradients = tape_model.gradient(total_mse, model.trainable_variables)
        optimizer.apply_gradients(
        zip(model_update_gradients, model.trainable_variables)
        )
  
      ending_time = time.time()
  
      print(f"Execution time of NN number {j+1}: ", ending_time-starting_time, flush=True)
  
      x_test = np.linspace(0, 2.5, 300)
  
      y_pred = model(x_test)
      
      theta0 = np.pi / 4  # Initial angle (rad)
      omega0 = 0.0        # Initial angular velocity (rad/s)
      t_0 = 0           # Start time (s)
      t_f = 2.5           # End time (s)
      dt = (t_f-t_0)/299           # Time step (s)
        
      # Number of steps
      n = int((t_f - t_0) / dt)
        
      # Arrays to store results

      y_kutta = np.zeros(n+1)
      omega = np.zeros(n+1)
        
      # Initial conditions
      y_kutta[0] = theta0
      omega[0] = omega0
        
      # Define the system of ODEs
      def f_theta(y_kutta, omega):
          return omega
        
      def f_omega(y_kutta, omega):
          return - (g / L) * np.sin(y_kutta)
        
      # Runge-Kutta 4th Order Method
      for i in range(n):
          k1_theta = dt * f_theta(y_kutta[i], omega[i])
          k1_omega = dt * f_omega(y_kutta[i], omega[i])
            
          k2_theta = dt * f_theta(y_kutta[i] + 0.5 * k1_theta, omega[i] + 0.5 * k1_omega)
          k2_omega = dt * f_omega(y_kutta[i] + 0.5 * k1_theta, omega[i] + 0.5 * k1_omega)
            
          k3_theta = dt * f_theta(y_kutta[i] + 0.5 * k2_theta, omega[i] + 0.5 * k2_omega)
          k3_omega = dt * f_omega(y_kutta[i] + 0.5 * k2_theta, omega[i] + 0.5 * k2_omega)
            
          k4_theta = dt * f_theta(y_kutta[i] + k3_theta, omega[i] + k3_omega)
          k4_omega = dt * f_omega(y_kutta[i] + k3_theta, omega[i] + k3_omega)
            
          y_kutta[i+1] = y_kutta[i] + (k1_theta + 2*k2_theta + 2*k3_theta + k4_theta) / 6
          omega[i+1] = omega[i] + (k1_omega + 2*k2_omega + 2*k3_omega + k4_omega) / 6
  
      val_error = 0
      for i in range(len(y_pred)):

          val_error += (y_pred[i] - y_kutta[i])**2

      val_error = val_error/len(y_pred)
      val_error = val_error[0]
  
      print('Test MSE Loss = {}'.format(val_error), flush=True)
  
    


    
    return error_arr, epoch_arr, [first_error, max_epoch, last_error, val_error]


##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################


def paralel_montecarlo_pinn_fraction(j,max_epochs=10000, n_nodes=7, seed=128):
    
    import os
    
    #Usamos el Keras 2.x, ya que es el único compatible con qml.qnn.KerasLayer
    
    os.environ["TF_USE_LEGACY_KERAS"] = "1"
    
    import numpy as np
    import tensorflow as tf
    #import matplotlib.pyplot as plt
    import pennylane as qml
    import time
    

    #Evitamos warnings sobre el cast complex a float
    
    tf.get_logger().setLevel('ERROR')
    
    #Forzamos a Tensorflow a trabajar con "float64" (desactivado)
    
    #tf.keras.backend.set_floatx('float64')
        
    
    num_domain = 30
    minval = 0
    maxval = 2
    
    def generate_uniform_data(num_domain, minval, maxval):
      data_init = tf.random_uniform_initializer(minval=minval, maxval=maxval, seed=seed)
      return tf.Variable(data_init(shape=[num_domain, 1]), dtype=tf.float32)
    
    x = generate_uniform_data(num_domain, minval, maxval)
    #gamma = 1

    ###
    
    def pde(tape1, x, y, j=0):
      dy_x = tape1.gradient(y, x)[:, j : j + 1]
      return dy_x + y*y
    
    def ic_0(x):
    
      return 1/2
    
    def ic_0_d(x):
    
      return 0
    
    
    x_0 = tf.Variable([[0]], shape=[1, 1], dtype=tf.float32)
    

    ###
    
    n_inputs = 1
    n_outputs = 1
    activation = 'tanh'
    
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units=4, activation=activation))
    model.add(tf.keras.layers.Dense(units=n_nodes, activation=activation))
    model.add(tf.keras.layers.Dense(units=n_outputs))
    
  
    epochs = max_epochs
    learning_rate = 0.01
    
    flag = False
    previous_mse = 0
    diff_tolerance = 0.00001
    tolerance = 0.0001
    max_count = 5
    count = 0
  
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
  
    error_arr = np.zeros(epochs//20 + 1)
  
    epoch_arr = np.linspace(0, epochs, epochs//20 + 1)
  
    while not flag:
      
      starting_time = time.time()
      
      for i in range(epochs + 1):
          
          
        with tf.GradientTape() as tape_model:
            

          ic_error = abs(model(x_0, training=True)[0][0] - ic_0(x_0))
        
                      
          with tf.GradientTape() as tape_pde:

              y = model(x, training=True)

              domain_error = pde(tape_pde, x, y)
  
          domain_mse = tf.math.reduce_mean(tf.math.square(domain_error), axis=0)
          ic_mse =  tf.math.square(ic_error)
          total_mse = domain_mse + 20*ic_mse
  
          #if i % 100 == 0:

            #print('Epoch: {}\t Total Loss = {}'.format(i, total_mse.numpy()[0]))
            #print('Domain Loss = {}'.format(domain_mse.numpy()[0]))
            #print('Initial and Periodical Condition Loss = {}'.format(20*ic_mse.numpy()))

  
          if i % 20 == 0:
  
            error_arr[i//20] = total_mse.numpy()[0]
  
          if i == 0:
  
            first_error = total_mse.numpy()[0]
  
            print('Montecarlo: {}\t Starting MSE Loss = {}'.format(j+1, first_error), flush=True)
  
  
  
  
        if abs(total_mse - previous_mse) < diff_tolerance and total_mse < tolerance:
  
          count += 1
  
          if count >= max_count:
  
            max_epoch = i
            last_error = total_mse.numpy()[0]
  
  
            print('Montecarlo: {}\t Final Epoch = {}\t Final MSE Loss = {}'.format(j+1, max_epoch, last_error), flush=True)
  
            flag = True
            break
  
  
        else:
  
          count = 0
  
        if i == epochs:
  
          max_epoch = i
          last_error = total_mse.numpy()[0]
  

          print('Montecarlo: {}\t Final Epoch = {}\t Final MSE Loss = {}'.format(j+1, max_epoch, last_error), flush=True)
  
          flag = True
          break
  
        previous_mse = total_mse
  
        model_update_gradients = tape_model.gradient(total_mse, model.trainable_variables)
        optimizer.apply_gradients(
        zip(model_update_gradients, model.trainable_variables)
        )
  
      ending_time = time.time()
  
      print(f"Execution time of NN number {j+1}: ", ending_time-starting_time, flush=True)
  
      x_test = np.linspace(0, 2, 300)
  
      y_pred = model(x_test)
      
      y_true = 1/(x_test+2)
  
      val_error = np.square(y_true - y_pred).mean()
  
      print('Test MSE Loss = {}'.format(val_error), flush=True)
  
    


    
    return error_arr, epoch_arr, [first_error, max_epoch, last_error, val_error]