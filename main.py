import VisualParafac as vp

if __name__ == '__main__':
    print("Hello!")
    default_way = "C:\\Users\\Tatiana\\Desktop\\ToBazhenov\\VD_DOM_Permafrost\\"
    data = vp.read_tensor(default_way)
    #data = vp.erase_Rayleigh(data, 24 // 2, 40 // 2)
    # Уточнить ширину линии в срезах по оси ox
    data = vp.erase_Rayleigh_Raman(data, 24 // 2, 40 // 2, 36 // 2, 36 // 2)
    vp.show_data(data,3,5,"DATA. NO RAYLEIGH & RAMAN")
    vp.show_components(data,2,2,4)
    vp.show_loadings(data,4)
