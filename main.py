import VisualParafac as vp

if __name__ == '__main__':
    print("Hello!")
    default_way = "C:\\Users\\Tatiana\\Desktop\\ToBazhenov\\VD_DOM_Permafrost\\"
    data = vp.read_tensor(default_way)
    data = vp.erase_Reyleigh(data,25//2,40//2)
    # Уточнить ширину линии в срезах по оси ox
    #data = erase_Reyleigh_Raman(data,25//2,40//2, 36//2,36//2)
    vp.show_data(data,3,5,"DATA",0,14)
    vp.show_components(data,2,2,4)
    vp.show_loadings(data,4)
