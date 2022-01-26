import re
import yaml

if __name__ == '__main__':

    c = re.compile(r"Iter (?P<iter>\b(\d|\.)*\b).*PSNR_1:0: (?P<psnr>\b(\d|\.)*\b).*SSIM:0: (?P<ssim>\b(\d|\.)*\b)")
    choose_dict = {}
    with open(r"runtime.log", 'r') as file:

        for i in file.readlines():
            match_result = c.search(i)
            if match_result is not None:
                match_dict = match_result.groupdict().copy()

                match_dict['iter'] = int(match_dict['iter'])
                match_dict['psnr'] = float(match_dict['psnr'])
                match_dict['ssim'] = float(match_dict['ssim'])
                if len(choose_dict) == 0 or choose_dict['psnr'] <= match_dict['psnr']:
                    choose_dict = match_dict
        with open("best_ckp.yaml", "w") as bestckp:
            yaml.dump(choose_dict, bestckp)
        print(f"export BEST_CKP=model_iters.{str(choose_dict['iter']).zfill(6)}.ckpt")
