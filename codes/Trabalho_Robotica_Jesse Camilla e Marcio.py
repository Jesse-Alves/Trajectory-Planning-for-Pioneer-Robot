#=========================================================================================================================================================
#===================================== TRABALHO FINAL DE ROBÓTICA MÓVEL ==================================================================================
#=========================================================================================================================================================

# ========
# EQUIPE  -  Jessé Alves, Camilla Bahia e Márcio Silva
# ========

import numpy as np
from numpy import sin,cos,tan,matmul,sqrt
import matplotlib.pyplot as plt
import time
import sys
import vrep

velocidadeAngularRodaDireita = 0.5
velocidadeAngularRodaEsquerda = 0.5
passo=0.05
Passo=0.05
LinearVelocity = 1

     
# SIMULACAO INICIAL PARA DEFINICAO DOS PONTOS INICIAIS E FINAIS  ============================================================================================  
        
vrep.simxFinish(-1) # just in case close all opened connections
clientID=vrep.simxStart('127.0.0.1',19999,True,True,5000,5)  # Connect to CoppeliaSim

if clientID!=-1:
        print ('Connected to remote API server')
       
        # enable the synchronous mode on the client:
        vrep.simxSynchronous(clientID,True)

        # start the simulation:
        vrep.simxStartSimulation(clientID,vrep.simx_opmode_blocking)


        ## Handle 
        errorCode,Robot =vrep.simxGetObjectHandle(clientID,'Pioneer_p3dx',vrep.simx_opmode_oneshot_wait)
    
        #Pose inicial
        errorCode,Posicao_robo_inicio = vrep.simxGetObjectPosition(clientID, Robot, vrep.sim_handle_parent, vrep.simx_opmode_oneshot_wait) # POSIÇÃO DO PIONEER
        print ("Posição Inicial = ",Posicao_robo_inicio)
        inicio = Posicao_robo_inicio
        
        #Pose final
        errorCode,targethandler=vrep.simxGetObjectHandle(clientID,'Target',vrep.simx_opmode_oneshot_wait) # TARGET
        errorCode,targetPosition = vrep.simxGetObjectPosition(clientID, targethandler, vrep.sim_handle_parent, vrep.simx_opmode_oneshot_wait) # POSIÇÃO DO TARGET
        print ("Posição Final = ",targetPosition)
        fim = targetPosition

else:
        print ('Failed connecting to remote API server')
        print ('Program ended')
        

# DEFININDO AS FUNCOES =====================================================================================================================================

#Classe do Robo Pionner
class Pionner():
    def __init__(self):
        # ====> Modelo Cinematico
        self.rodaDiametro = (195)/1000 
        self.raio = self.rodaDiametro/2
        self.rodaDireitaRaio = self.rodaDiametro/2 
        self.rodaEsquerdaRaio = self.rodaDiametro/2 
        self.distanciaRodaEixo = ((381)/1000)/2 
        self.velocidadeAngularRodaDireita = 0.5
        self.velocidadeAngularRodaEsquerda = 0.5
        self._posicaoRoboXYtheta = np.array(inicio).reshape(3,1)

    @property
    def velocidadeAngularRodas(self,vd,ve):
        #return np.array([self.velocidadeAngularRodaDireita,self.velocidadeAngularRodaEsquerda]).reshape((2,1))
        self.velocidadeAngularRodaDireita = vd
        self.velocidadeAngularRodaEsquerda = ve
        
    @property
    def posicaoRobo(self) -> np.ndarray:
        return self._posicaoRoboXYtheta

    @property
    def velocidadeAngularRodas(self) -> np.ndarray:
        return np.array([self.velocidadeAngularRodaDireita,
                        self.velocidadeAngularRodaEsquerda]).reshape((2,1))

    @property
    def velocidadeLinearRodas(self) -> float:
        return self.velocidadeAngularRodaDireita * self.raio, self.velocidadeAngularRodaEsquerda * self.raio
    @property
    def posicaoX(self) -> float:
        return float(self._posicaoRoboXYtheta[0])
    @property
    def posicaoY(self) -> float:
        return float(self._posicaoRoboXYtheta[1])
    @property  
    def rotacaoTheta(self) -> float:
        return float(self._posicaoRoboXYtheta[2])

    @posicaoRobo.setter
    def posicaoRobo(self,Posicao:np.ndarray) -> np.ndarray:
        self._posicaoRoboXYtheta = Posicao.reshape(3,1)

    @velocidadeAngularRodas.setter
    def velocidadeAngularRodas(self,velocidadesAngularesRodas):
        self.velocidadeAngularRodaDireita, self.velocidadeAngularRodaEsquerda = velocidadesAngularesRodas
        
        
    def posicaoGlobal(self):
        rotacao = float(self._posicaoRoboXYtheta[2])
        posicaoGlobal = np.array([cos(rotacao),   -sin(rotacao),  0, 
                            sin(rotacao),   cos(rotacao),   0,
                             0           ,0,                 1]).reshape((3,3))

        coefposicaoGlobal =  np.array([1,0,
                                       0,0,
                                       0,1]).reshape((3,2))

        return np.matmul(posicaoGlobal,coefposicaoGlobal)
    

    def velocidadeRobo(self):

        rD = self.rodaDireitaRaio
        rE = self.rodaEsquerdaRaio
        L =  self.distanciaRodaEixo

        RelacaoVelocidadeRodaRobo = np.array([rD/2, rE/2, 
                                            (rD/(2*L)), (-rE/(2*L))]).reshape((2,2))        
        velocidadeAngularRoda = np.array([self.velocidadeAngularRodaDireita,self.velocidadeAngularRodaEsquerda]).reshape((2,1))
        velocidadeRobo = np.matmul(RelacaoVelocidadeRodaRobo,velocidadeAngularRoda)
        return velocidadeRobo


    def calculate_new_position(self,Passo):
        velocidadeGlobalRobo = np.matmul(self.posicaoGlobal(),self.velocidadeRobo()) #Botar os parenteses
        deslocamentoRobo = velocidadeGlobalRobo*Passo
        self.posicaoRobo = self._posicaoRoboXYtheta + deslocamentoRobo
        return self.posicaoRobo
      


def conv_velocidades(v,w):
    vel_linear = np.array([v,w])
    #Dimenções Pionner
    r = (195)/2000
    L = (381)/2000
    
    Dim_Matriz = np.array([1/r,L/r,1/r,-L/r]).reshape(2,2)
    vel_rodas = np.matmul(Dim_Matriz,vel_linear)
    return vel_rodas



def NonHolonomicTrajectory(InitialPose,FinalPose,LinearVelocity,Step = 0.05):
    jxi = float(InitialPose[0])
    jyi = float(InitialPose[1])
    jOi = float(InitialPose[2])

    jxf = float(FinalPose[0])
    jyf = float(FinalPose[1])
    jOf = float(FinalPose[2])

    deltax = jxf - jxi
    deltay = jyf - jyi
    di = tan(jOi)
    df = tan(jOf)

    a0 = jxi
    a1 = deltax
    a2 = 0
    a3 = deltax - a2 - a1
    b0 = jyi
    b1 = di*a1
    b2 = 3*(deltay - df*deltax) + df*a2 - 2*(di-df)*a1
    b3 = 3*df*deltax-2*deltay-df*a2 - (2*df-di)*a1

    LambdaParameter = np.array([0]).reshape((1,1))
    lambda_k = float(LambdaParameter[0])
    T = Step
    v_k = LinearVelocity

    while lambda_k < 1:
        dx_k = matmul([a1, 2*a2, 3*a3],[1,lambda_k,lambda_k**2])
        dy_k = matmul([b1, 2*b2, 3*b3], [1,lambda_k,lambda_k**2])

        dlambda_k = v_k/(sqrt(dx_k**2 + dy_k**2))
        lambda_k = lambda_k+ dlambda_k*T
        LambdaParameter = np.append(LambdaParameter,lambda_k)

    StepsNumber = LambdaParameter.shape
    StepsNumber = int(StepsNumber[0])
    LambdaParameter = LambdaParameter.reshape((1,StepsNumber))

    lambda_matrix_cubic = np.array([np.ones((1,StepsNumber)),
                             LambdaParameter, 
                             LambdaParameter**2,
                             LambdaParameter**3]).reshape((4,StepsNumber))

    lambda_matrix_square = np.array([np.ones((1,StepsNumber)),
                             LambdaParameter, 
                             LambdaParameter**2]).reshape((3,StepsNumber))

    dx_dlambda = matmul([a1,2*a2,3*a3],lambda_matrix_square)
    dy_dlambda = matmul([b1,2*b2,3*b3],lambda_matrix_square)

    PathX = matmul([a0, a1, a2, a3],lambda_matrix_cubic)
    PathY = matmul([b0, b1, b2, b3],lambda_matrix_cubic)
    PathO = np.divide(dy_dlambda,dx_dlambda)
    PathO = np.arctan(PathO)

    dx_dt = np.gradient(PathX,Step)
    dy_dt = np.gradient(PathY,Step)
    dO_dt = np.gradient(PathO,Step)

    v_t = sqrt(dx_dt**2 + dy_dt**2)
    w_t = dO_dt

    return v_t,w_t



class Pionner_Modelo(Pionner):
    def __init__(self) -> None:
        super().__init__()

    @property
    def posicaoGlobal(self) -> np.ndarray:
        rotacao = self.rotacaoTheta
        posicaoGlobal = np.array([cos(rotacao),   -sin(rotacao),  0, 
                            sin(rotacao),   cos(rotacao),   0,
                             0           ,0,                 1]).reshape((3,3))

        coefposicaoGlobal =  np.array([1,0,
                                       0,0,
                                       0,1]).reshape((3,2))

        return np.matmul(posicaoGlobal,coefposicaoGlobal)

    @property
    def velocidadeRobo(self) -> np.ndarray:

        rD = self.rodaDireitaRaio
        rE = self.rodaEsquerdaRaio
        L =  self.distanciaRodaEixo

        RelacaoVelocidadeRodaRobo = np.array([rD/2, rE/2, 
                                            (rD/(2*L)), (-rE/(2*L))]).reshape((2,2))
        
        velocidadeAngularRoda = self.velocidadeAngularRodas
        velocidadeRobo = np.matmul(RelacaoVelocidadeRodaRobo,velocidadeAngularRoda)
        return velocidadeRobo

    def calculaNovaPosicaoRobo(self, Passo: float = 0.05):
        velocidadeGlobalRobo = np.matmul(self.posicaoGlobal,self.velocidadeRobo)
        deslocamentoRobo = velocidadeGlobalRobo*Passo
        self.posicaoRobo = self.posicaoRobo + deslocamentoRobo
        return self.posicaoRobo
    

 
  
#=====================================================================================================================================================
#======================= INICIANDO O PROGRAMA ========================================================================================================
#=====================================================================================================================================================

inicio = [0,0,0]
fim =  [7,12,0]
passo = 0.05

v, w = NonHolonomicTrajectory(inicio,fim,LinearVelocity = 1,Step = passo)
t = passo*np.arange(0,len(v))


vel_rodas = conv_velocidades(v,w)
W_direita = vel_rodas[0,:]  #VelocidadeAngularDireita 
W_esquerda = vel_rodas[1,:]




# SIMULACAO  PARA O ROBO CHEGAR AO TARGET   ===========================================================================================================
vrep.simxFinish(-1) # just in case, close all opened connections
clientID=vrep.simxStart('127.0.0.1',19997,True,True,5000,5) # Connect to CoppeliaSim
if clientID!=-1:
    print ('Connected to remote API server')

    # enable the synchronous mode on the client:
    vrep.simxSynchronous(clientID,True)

    # start the simulation:
    vrep.simxStartSimulation(clientID,vrep.simx_opmode_blocking)

    ## Handle 
    returnCode,Robot = vrep.simxGetObjectHandle(clientID,'Pioneer_p3dx',vrep.simx_opmode_blocking)
    returnCode,left_Motor= vrep.simxGetObjectHandle(clientID,'Pioneer_p3dx_leftMotor',vrep.simx_opmode_blocking)
    returnCode,right_Motor= vrep.simxGetObjectHandle(clientID,'Pioneer_p3dx_rightMotor',vrep.simx_opmode_blocking)
    returnCode,front_Sensor =vrep.simxGetObjectHandle(clientID,'Pioneer_p3dx_ultrasonicSensor5',vrep.simx_opmode_blocking)
    returnCode,camera= vrep.simxGetObjectHandle(clientID,'Vision_sensor',vrep.simx_opmode_blocking)
    returnCode,resolution,image = vrep.simxGetVisionSensorImage(clientID,camera,1,vrep.simx_opmode_streaming)

    T = passo # 0.05
    N = len(W_direita)

    errorCode,Real_Robot_position = vrep.simxGetObjectPosition(clientID, Robot, vrep.sim_handle_parent, vrep.simx_opmode_oneshot_wait) # POSIÇÃO DO PIONEER
   
    P = np.array(Real_Robot_position).reshape((3,1))
    PosicaoInicial = P
    PosicaoRealHistorico = P - PosicaoInicial
    PosicaoRealHistorico =  PosicaoInicial

    for i in range(3):
        vrep.simxSynchronousTrigger(clientID)
        time.sleep(1)


    for i in range(0,N-1):
        
        vrep.simxSynchronousTrigger(clientID)

        #Velocidade de referencia
        v_motor_l= W_esquerda[i]
        v_motor_r= W_direita[i]

        returnCode=vrep.simxSetJointTargetVelocity(clientID,left_Motor,v_motor_l,vrep.simx_opmode_blocking)
        returnCode=vrep.simxSetJointTargetVelocity(clientID,right_Motor,v_motor_r,vrep.simx_opmode_blocking)

        returnCode,Real_Robot_position = vrep.simxGetObjectPosition(clientID,Robot,-1,vrep.simx_opmode_blocking)
        #P = np.array(Real_Robot_position).reshape((3,1)) - PosicaoInicial
        P = np.array(Real_Robot_position).reshape((3,1)) 
        PosicaoRealHistorico = np.append(PosicaoRealHistorico,P,axis=1)
        b = "Simulando " + str((i/(N-1))*100) + '%'
        #print (b)



    # stop the simulation:
    vrep.simxStopSimulation(clientID,vrep.simx_opmode_blocking)
    print()
    print('Simulation Stopped')

    # Now close the connection to CoppeliaSim:
    vrep.simxFinish(clientID)
    print('Simulation Closed')


# DEFININDO A TRAJETORIA   ===========================================================================================================================

#passo = 0.05
#v, w = NonHolonomicTrajectory(inicio,fim,LinearVelocity = 1,Step = passo)
#t = passo*np.arange(0,len(v))
# Valores Inicias
RoboPi = Pionner()

# Valores Inicias
PoseRobotModel = np.array(inicio).reshape(3,1)
VelocidadeRoboHistorico = np.array([0,0]).reshape((2,1))
itera = vel_rodas.shape[1]

#for i in range(0,itera):
#    RoboPi.velocidadeAngularRodas(vel_rodas[0,i],vel_rodas[1,i])
#    NewPose = RoboPi.calculate_new_position(passo)    
#    VelocidadeNovaRobo = RoboPi.velocidadeRobo()    
#    VelocidadeRoboHistorico = np.append(VelocidadeRoboHistorico,VelocidadeNovaRobo,axis=1)    
#    PoseRobotModel = np.append(PoseRobotModel,NewPose,axis=1)
    

# Valores Inicias
robot = Pionner_Modelo()
PoseRobo = robot.posicaoRobo

# SIMULAÇÃO DO MODELO CINEMATICO
VelocidadeRoboHistorico = np.array([0,0]).reshape((2,1))
itera = vel_rodas.shape[1]
for i in range(0,itera):
    robot.velocidadeAngularRodas = [vel_rodas[0,i],vel_rodas[1,i]]
    PosicaoNova = robot.calculaNovaPosicaoRobo(Passo=0.05)
    VelocidadeNovaRobo = robot.velocidadeRobo
    VelocidadeRoboHistorico = np.append(VelocidadeRoboHistorico,VelocidadeNovaRobo,axis=1)
    PoseRobo = np.append(PoseRobo,PosicaoNova,axis=1)

    
PosicaoX = PoseRobo[0,:]
PosicaoY = PoseRobo[1,:]
Angulo = PoseRobo[2,:]


PosicaoRealX = PosicaoRealHistorico[0,:]
PosicaoRealY = PosicaoRealHistorico[1,:]
AnguloReal = PosicaoRealHistorico[2,:]

plt.figure(0)
plt.plot(PosicaoRealX,PosicaoRealY,color='red', label="Ground Truth")
plt.plot(PosicaoX,PosicaoY,color='blue', label="Modelo Cinemático")
Title = 'Trajetória'
plt.title(Title)
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('square')
plt.legend()


AngularVelocities = conv_velocidades(v,w)
velocidadeAngularRodaDireita = AngularVelocities[0,:]
velocidadeAngularRodaEsquerda = AngularVelocities[1,:]


plt.figure(1)
plt.plot(t,velocidadeAngularRodaDireita,color='red',label = "Velocidade Angular Roda Direita [Vd]")
plt.plot(t,velocidadeAngularRodaEsquerda,color='blue',label = "Velocidade Angular Roda Esquerda [Ve]")
Title = 'Velocidades'
plt.title(Title)
plt.xlabel('s')
plt.ylabel('v')
plt.axis('square')
plt.legend()

