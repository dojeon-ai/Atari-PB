data_dir='your-directory-here'

games='AirRaid Amidar Asteroids Atlantis BankHeist BattleZone Berzerk Bowling Boxing Breakout Carnival Centipede ChopperCommand CrazyClimber DemonAttack DoubleDunk ElevatorAction Enduro FishingDerby Freeway Frostbite Gopher Gravitar Hero IceHockey Jamesbond Kangaroo Krull KungFuMaster MontezumaRevenge MsPacman NameThisGame Phoenix Pitfall PrivateEye Qbert RoadRunner Robotank Skiing Solaris SpaceInvaders StarGunner Tennis TimePilot Tutankham UpNDown VideoPinball WizardOfWor YarsRevenge Zaxxon Alien Assault Asterix BeamRider JourneyEscape Pong Pooyan Riverraid Seaquest Venture'
runs='1 2'
ckpts='1 2 3 4 5 6 7 8 9 10'
files='action observation reward terminal'
for g in ${games[@]}; do
  mkdir -p "${data_dir}/${g}"
  for f in ${files[@]}; do
    for c in ${ckpts[@]}; do
      for r in ${runs[@]}; do
          if [ ! -f "${data_dir}/${g}/${f}_${r}_${c}.gz" ]; then
            gsutil cp "gs://atari-replay-datasets/dqn/${g}/${r}/replay_logs/\$store\$_${f}_ckpt.${c}.gz" "${data_dir}/${g}/${f}_${r}_${c}.gz"
          fi;
      done;
    done;
  done;
done;