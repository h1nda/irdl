#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2022/2023
#############################################################################
###
### Домашно задание 2
###
#############################################################################

import numpy as np

#############################################################

def sigmoid(x):
    return 1/(1+np.exp(-x))

def lossAndGradient(u_w, Vt, W):
    ###  Векторът u_w е влагането на целевата дума. shape(u_w) = M.
    ###  Матрицата Vt представя влаганията на контекстните думи. shape(Vt) = (n+1)xM.
    ###  Първият ред на Vt е влагането на коректната контекстна дума, а
    ###  следващите n реда са влаганията на извадката от негативни контекстни думи
	###  Матрицата W е параметър с теглата на квадратичната форма. shape(W) = MxM.
    ###
    ###  функцията връща J -- загубата в тази точка;
    ###                  du_w -- градиентът на J спрямо u_w;
    ###                  dVt --  градиентът на J спрямо Vt.
    ###                  dW --  градиентът на J спрямо W.
    #############################################################################
    #### Начало на Вашия код. На мястото на pass се очакват 7-15 реда
    delta = np.zeros(shape=Vt.shape[0])
    delta[0] = 1
    vsigmoid = np.vectorize(sigmoid) 
    VtW = np.matmul(Vt, W) # n+1 X M
    VtWu_w = np.matmul(VtW, u_w)
    Wu_w = np.matmul(W, u_w)
    sig_vec = vsigmoid(VtWu_w) - delta
    du_w = np.einsum('i,ij->j', sig_vec, VtW)
    dVt = np.outer(sig_vec, Wu_w)
    dW = np.einsum('i,ij,k->jk', sig_vec, Vt, u_w)
    
    sign = np.full(Vt.shape[0], fill_value=-1)
    sign[0] = 1
    J = -np.sum(np.log(vsigmoid(sign*VtWu_w)))

    #### Край на Вашия код
    #############################################################################

    return J, du_w, dVt, dW


def lossAndGradientCumulative(u_w, Vt, W):
    ###  Изчисляване на загуба и градиент за цяла партида
    ###  Тук за всяко от наблюденията се извиква lossAndGradient
    ###  и се акумулират загубата и градиентите за S-те наблюдения
    Cdu_w = []
    CdVt = []
    CJ = 0
    CdW = 0
    S = u_w.shape[0]
    for i in range(S):
        J, du_w, dVt, dW = lossAndGradient(u_w[i],Vt[i],W)
        Cdu_w.append(du_w/S)
        CdVt.append(dVt/S)
        CdW += dW/S
        CJ += J/S
    return CJ, Cdu_w, CdVt, CdW


def lossAndGradientBatched(u_w, Vt, W):
    ###  Изчисляване на загуба и градиент за цяла партида.
    ###  Тук едновременно се изчислява загубата и градиентите за S наблюдения.
    ###  Матрицата u_w представя влаганията на целевите думи и shape(u_w) = SxM.
    ###  Тензорът Vt представя S матрици от влагания на контекстните думи и shape(Vt) = Sx(n+1)xM.
   	###  Матрицата W е параметър с теглата на квадратичната форма. shape(W) = MxM.
    ###  Във всяка от S-те матрици на Vt в първия ред е влагането на коректната контекстна дума, а
    ###  следващите n реда са влаганията на извадката от негативни контекстни думи.
    ###
    ###  Функцията връща J -- загубата за цялата партида;
    ###                  du_w -- матрица с размерност SxM с градиентите на J спрямо u_w за всяко наблюдение;
    ###                  dVt --  с размерност Sx(n+1)xM -- S градиента на J спрямо Vt.
    ###                  dW --  матрица с размерност MxM -- партидния градиент на J спрямо W.
    #############################################################
    ###  От вас се очаква вместо да акумулирате резултатите за отделните наблюдения,
    ###  да използвате тензорни операции, чрез които наведнъж да получите
    ###  резултата за цялата партида. Очаква се по този начин да получите над 2 пъти по-бързо изпълнение.
    #############################################################

    #############################################################################
    #### Начало на Вашия код. На мястото на pass се очакват 10-20 реда
    delta = np.zeros(shape=Vt.shape[1]) # delta.shape = (n+1)
    delta[0] = 1
    vsigmoid = np.vectorize(sigmoid) 

    # u_w sa redovete!!!!
    S = u_w.shape[0]
    # shape(Vt) = Sx(n+1)xM.
    # shape(u_w) = SxM.
    # shape(W) = MxM.
    VtW = np.einsum("sij,jk->sik",Vt, W) #VtW.shape = Sx(n+1)xM     # Sx(n+1)xM, SxM
    VtWu_w = np.einsum("sij,jk,sk->si", Vt, W, u_w) # VtWu_w.shape = Sx(n+1)
    Wu_w = np.einsum("ij,sj->si", W, u_w) # Wu_w.shape = SxM

    sig_matrix = vsigmoid(VtWu_w) - delta # sig_matrix.shape = Sx(n+1)

    du_w = np.einsum("si,sij->sj", sig_matrix, VtW) / S 
    dVt = np.einsum("si,sj->sij",sig_matrix, Wu_w) / S 
    dW = np.einsum("si,sij,sk->jk",sig_matrix,Vt,u_w) / S

    sign = np.full(Vt.shape[1], fill_value=-1)
    sign[0] = 1
    J = np.sum(np.log(vsigmoid(np.einsum("ij,j->ij",VtWu_w,sign))),axis=0)
    J = -np.sum(J)/S
    #### Край на Вашия код
    ############################################################################
    return J, du_w, dVt, dW