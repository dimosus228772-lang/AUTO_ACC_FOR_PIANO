#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Безопасный запуск:
- Игнорирует Midi Through, выбирает первое реальное устройство (или укажете сами)
- Тест приёма (попросит нажать клавишу) и тест отправки (сыграет тихий C-dur)
- По Ctrl+C/ошибке гарантирует All Notes Off
"""

import sys, time
import mido
from mido import Message
from auto_accomp_pro import AutoAccompPro

def pick_ports(user_in=None, user_out=None):
    if user_in and user_out:
        return user_in, user_out
    ins=[p for p in mido.get_input_names() if "Midi Through" not in p]
    outs=[p for p in mido.get_output_names() if "Midi Through" not in p]
    if not ins or not outs:
        print("[ERROR] Не найдено MIDI-устройство. Подключите пианино."); sys.exit(1)
    print(f"[INFO] Выбраны порты:\n  Вход:  {ins[0]}\n  Выход: {outs[0]}")
    return ins[0], outs[0]

def test_ports(in_port, out_port):
    print("[INFO] Проверка портов...")
    ok_in=False
    with mido.open_input(in_port) as inp:
        print("[TEST] Нажмите любую клавишу на пианино (5с)...")
        t0=time.time()
        while time.time()-t0<5:
            for msg in inp.iter_pending():
                if msg.type=='note_on' and msg.velocity>0:
                    print(f"[OK] Вход работает (note {msg.note})."); ok_in=True; break
            if ok_in: break
            time.sleep(0.01)
    if not ok_in: print("[WARN] Не увидел событий на входе.")

    try:
        with mido.open_output(out_port) as outp:
            notes=[60,64,67]
            for n in notes: outp.send(Message('note_on', note=n, velocity=40, channel=0))
            time.sleep(0.4)
            for n in notes: outp.send(Message('note_off', note=n, velocity=0, channel=0))
        print("[OK] Выход работает (сыгран тестовый аккорд).")
    except Exception as e:
        print(f"[ERROR] Ошибка выхода: {e}"); sys.exit(1)

def main():
    in_port, out_port = pick_ports()
    test_ports(in_port, out_port)
    acc = AutoAccompPro(in_port=in_port, out_port=out_port,
                        analyze_secs=15.0, program=0, hud=True, verbose=True)
    try:
        acc.run()
    except KeyboardInterrupt:
        print("\n[INFO] Прервано пользователем.")
    finally:
        try:
            acc.all_notes_off()
        except Exception:
            # на случай, если порты уже закрыты
            pass
        print("[INFO] Все ноты выключены. Завершение.")

if __name__=='__main__':
    main()
