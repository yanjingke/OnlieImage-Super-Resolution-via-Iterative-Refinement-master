from  clip import clip
file_object2 = open(r"F:\data\CelebAMask-HQ\CelebAMask-HQ-at.txt",'r')
typest="5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young"
types=typest.split()
lines = file_object2.readlines()

for line in lines:
      print(line.split()[0])
      types_nb=line.split()[1:]
      str_ty = ""
      for i in range(0,len(types_nb)):
            if int(types_nb[i])==1:
                  str_ty=str_ty+types[i]+"."

      text = clip.tokenize( str_ty)

      print( text.shape )